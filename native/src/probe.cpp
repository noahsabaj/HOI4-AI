// Transport/capability probe. Deliberately NOT an in-process HOI4 adapter.
#include <windows.h>
#include <sddl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>
#ifdef HOI4_IN_PROCESS_PROBE
#include <bcrypt.h>
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t max_message = 4 * 1024 * 1024;

#ifdef HOI4_IN_PROCESS_PROBE
std::string executable_hash;
DWORD loader_thread = 0;
bool supported_build = false;
std::string hash_executable() {
    wchar_t path[32768];
    const auto length = GetModuleFileNameW(nullptr, path, 32768);
    if (!length || length >= 32768) throw std::runtime_error("cannot identify game executable");
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot read game executable");
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::string result;
    try {
        if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
            BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0)
            throw std::runtime_error("cannot create SHA-256 context");
        std::vector<unsigned char> buffer(65536);
        DWORD read = 0;
        while (true) {
            if (!ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr))
                throw std::runtime_error("cannot hash game executable");
            if (!read) break;
            if (BCryptHashData(hash, buffer.data(), read, 0) < 0)
                throw std::runtime_error("SHA-256 update failed");
        }
        unsigned char digest[32];
        if (BCryptFinishHash(hash, digest, 32, 0) < 0) throw std::runtime_error("SHA-256 failed");
        const char* digits = "0123456789abcdef";
        for (const auto byte : digest) {
            result += digits[byte >> 4];
            result += digits[byte & 15];
        }
    } catch (...) {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        CloseHandle(file);
        throw;
    }
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    CloseHandle(file);
    return result;
}
#endif

struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
};

struct Descriptor {
    PSECURITY_DESCRIPTOR value = nullptr;
    Descriptor() {
        Handle token;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token.value))
            throw std::runtime_error("OpenProcessToken failed");
        DWORD size = 0;
        GetTokenInformation(token.value, TokenUser, nullptr, 0, &size);
        std::vector<unsigned char> data(size);
        if (!GetTokenInformation(token.value, TokenUser, data.data(), size, &size))
            throw std::runtime_error("GetTokenInformation failed");
        LPWSTR sid = nullptr;
        if (!ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(data.data())->User.Sid, &sid))
            throw std::runtime_error("ConvertSidToStringSid failed");
        std::wstring sddl = L"D:P(A;;GA;;;" + std::wstring(sid) + L")";
        LocalFree(sid);
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1,
                                                                  &value, nullptr))
            throw std::runtime_error("pipe security descriptor failed");
    }
    ~Descriptor() { if (value) LocalFree(value); }
};

void transfer(HANDLE pipe, void* buffer, DWORD size, bool write, Clock::time_point deadline) {
    DWORD done = 0;
    auto* bytes = static_cast<unsigned char*>(buffer);
    while (done < size) {
        if (Clock::now() >= deadline) throw std::runtime_error("client deadline exceeded");
        DWORD count = 0;
        const BOOL ok = write ? WriteFile(pipe, bytes + done, size - done, &count, nullptr)
                              : ReadFile(pipe, bytes + done, size - done, &count, nullptr);
        if (!ok && GetLastError() != ERROR_NO_DATA && GetLastError() != ERROR_PIPE_LISTENING)
            throw std::runtime_error("pipe disconnected");
        done += count;
        if (!count) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

json respond(const json& request) {
    json result = {{"version", 1}, {"id", request.value("id", json(nullptr))}, {"ok", false}};
    if (!request.is_object() || request.value("version", json()) != 1 || !request.contains("id") ||
        !request["id"].is_number_integer() || request["id"].get<std::int64_t>() <= 0 ||
        !request.contains("payload") || !request["payload"].is_object()) {
        result["error"] = "invalid_protocol";
        return result;
    }
    if (request.value("method", std::string()) == "hello") {
        result["ok"] = true;
        result["result"] = {
            {"backend", "diagnostic"}, {"process_id", GetCurrentProcessId()},
            {"capabilities", {
                {"player_view", false}, {"normal_orders", false}, {"simulation_thread_dispatch", false},
                {"independent_country_control", false}, {"country_ai_suppression", false},
                {"reset", false}, {"outcomes", false}, {"human_input_isolation", false},
                {"accepted_command_stream", false}}},
            {"reason", "No verified engine adapter exists for the installed HOI4 build."}
        };
#ifdef HOI4_IN_PROCESS_PROBE
        result["result"]["backend"] = "hoi4_in_process_probe";
        result["result"]["executable_sha256"] = executable_hash;
        result["result"]["supported_build"] = supported_build;
        result["result"]["lua_loader_thread_id"] = loader_thread;
        result["result"]["pipe_thread_id"] = GetCurrentThreadId();
        result["result"]["reason"] = "Native module loaded inside HOI4; engine observation/order adapter is unimplemented.";
#endif
    } else {
        result["error"] = "engine_adapter_unavailable";
    }
    return result;
}

void serve_client(HANDLE pipe, bool persistent) {
    for (int request_number = 0; persistent || request_number < 1024; ++request_number) {
        // A persistent host tolerates an idle client between requests (a learner
        // update can take minutes); a started frame must still finish promptly.
        std::uint32_t size = 0;
        transfer(pipe, &size, sizeof(size), false,
                 Clock::now() + (persistent ? std::chrono::seconds(3600) : std::chrono::seconds(5)));
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        if (!size || size > max_message) throw std::runtime_error("message size rejected");
        std::string body(size, '\0');
        transfer(pipe, body.data(), size, false, deadline);
        const auto parsed = json::parse(body, nullptr, false);
        const auto reply = parsed.is_object() ? respond(parsed)
            : json{{"version", 1}, {"id", nullptr}, {"ok", false}, {"error", "invalid_json"}};
        std::string response = reply.dump();
        size = static_cast<std::uint32_t>(response.size());
        transfer(pipe, &size, sizeof(size), true, deadline);
        transfer(pipe, response.data(), size, true, deadline);
    }
}

int run_server(const std::wstring& requested_name, bool persistent) {
    try {
        std::wstring name(requested_name);
        if (!name.starts_with(L"hoi4-arena-") || name.size() > 91 || name.size() < 12 ||
            !std::all_of(name.begin(), name.end(), [](wchar_t c) {
                return (c >= L'a' && c <= L'z') || (c >= L'A' && c <= L'Z') ||
                       (c >= L'0' && c <= L'9') || c == L'-' || c == L'_';
            })) throw std::runtime_error("invalid local pipe name");
        name = L"\\\\.\\pipe\\" + name;
        Descriptor descriptor;
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), descriptor.value, FALSE};
        Handle pipe;
        pipe.value = CreateNamedPipeW(name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS,
            1, 65536, 65536, 0, &security);
        if (pipe.value == INVALID_HANDLE_VALUE) throw std::runtime_error("CreateNamedPipe failed");
        std::cout << "ready: diagnostic transport only" << std::endl;
        // The standalone probe serves one client. Inside the game the host must
        // survive a crashed or restarted Python client, so it listens again.
        do {
            const auto connect_deadline = Clock::now() + std::chrono::seconds(120);
            while (!ConnectNamedPipe(pipe.value, nullptr)) {
                const auto error = GetLastError();
                if (error == ERROR_PIPE_CONNECTED) break;
                if (error != ERROR_PIPE_LISTENING && error != ERROR_NO_DATA)
                    throw std::runtime_error("ConnectNamedPipe failed");
                if (!persistent && Clock::now() >= connect_deadline) throw std::runtime_error("no client connected");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            try {
                serve_client(pipe.value, persistent);
            } catch (const std::exception&) {
                if (!persistent) throw;
            }
            DisconnectNamedPipe(pipe.value);
        } while (persistent);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

#ifdef HOI4_IN_PROCESS_PROBE
DWORD WINAPI serve_probe(LPVOID) {
    return static_cast<DWORD>(run_server(L"hoi4-arena-game-" + std::to_wstring(GetCurrentProcessId()), true));
}

// Lua's native callback ABI is int(lua_State*). This probe deliberately does
// not call Lua APIs or read its opaque state, so no private Lua ABI is assumed.
extern "C" __declspec(dllexport) int load_probe(void*) {
    static volatile LONG started = 0;
    if (InterlockedCompareExchange(&started, 1, 0) != 0) return 0;
    try {
        loader_thread = GetCurrentThreadId();
        executable_hash = hash_executable();
        supported_build = executable_hash == ARENA_SUPPORTED_EXECUTABLE_SHA256;
        // Keep the module alive for the pipe worker; never run work in DllMain.
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(&load_probe), &module)) return 0;
        Handle worker;
        worker.value = CreateThread(nullptr, 0, serve_probe, nullptr, 0, nullptr);
    } catch (...) {
        // No C++ exception may cross the game/Lua C boundary.
    }
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) DisableThreadLibraryCalls(instance);
    return TRUE;
}
#else
int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        std::cerr << "usage: hoi4_bridge_probe hoi4-arena-NAME\n";
        return 2;
    }
    return run_server(argv[1], false);
}
#endif
