#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string>
#include <map>
#include <vector>
#include <memory>

namespace py = pybind11;

class FastOCR {
private:
    std::unique_ptr<tesseract::TessBaseAPI> ocr;
    std::map<std::string, std::vector<int>> regions;
    int screen_width;
    int screen_height;

public:
    FastOCR(int width = 3840, int height = 2160) : screen_width(width), screen_height(height) {
        // Initialize Tesseract
        ocr = std::make_unique<tesseract::TessBaseAPI>();
        if (ocr->Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY)) {
            throw std::runtime_error("Could not initialize Tesseract");
        }

        // Optimize for speed
        ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
        ocr->SetVariable("tessedit_do_invert", "0");
        ocr->SetVariable("load_system_dawg", "false");
        ocr->SetVariable("load_freq_dawg", "false");

        // Initialize regions for 4K (scale from 1920x1080 base)
        float scale_x = (float)width / 1920.0f;
        float scale_y = (float)height / 1080.0f;

        // Base regions at 1920x1080
        std::map<std::string, std::vector<int>> base_regions = {
            {"date", {860, 10, 200, 40}},
            {"political_power", {250, 10, 100, 40}},
            {"factories", {1700, 10, 200, 50}},
            {"manpower", {400, 10, 150, 40}},
            {"division_count", {1700, 900, 200, 50}}
        };

        // Scale to actual resolution
        for (const auto& [name, coords] : base_regions) {
            regions[name] = {
                (int)(coords[0] * scale_x),
                (int)(coords[1] * scale_y),
                (int)(coords[2] * scale_x),
                (int)(coords[3] * scale_y)
            };
        }
    }

    ~FastOCR() {
        if (ocr) {
            ocr->End();
        }
    }

    std::string extract_region_internal(uint8_t* data, int x, int y, int w, int h) {
        // Create Pix from image region
        Pix* pix = pixCreate(w, h, 32);

        // Copy region data
        for (int py = 0; py < h; py++) {
            for (int px = 0; px < w; px++) {
                int src_idx = ((y + py) * screen_width + (x + px)) * 3;
                uint8_t r = data[src_idx];
                uint8_t g = data[src_idx + 1];
                uint8_t b = data[src_idx + 2];
                pixSetRGBPixel(pix, px, py, r, g, b);
            }
        }

        // Preprocess for better OCR
        Pix* gray = pixConvertRGBToGray(pix, 0.299f, 0.587f, 0.114f);
        pixDestroy(&pix);

        // Apply threshold
        Pix* binary = pixThresholdToBinary(gray, 128);
        pixDestroy(&gray);

        // OCR
        ocr->SetImage(binary);
        char* text = ocr->GetUTF8Text();
        std::string result(text ? text : "");
        delete[] text;

        pixDestroy(&binary);

        // Clean up result
        result.erase(result.find_last_not_of(" \n\r\t") + 1);
        return result;
    }

    std::map<std::string, std::string> extract_all_text(py::array_t<uint8_t> image) {
        std::map<std::string, std::string> results;

        auto buf = image.request();
        if (buf.ndim != 3) {
            throw std::runtime_error("Image must be 3D (height, width, channels)");
        }

        uint8_t* data = static_cast<uint8_t*>(buf.ptr);

        // Extract each region
        for (const auto& [name, coords] : regions) {
            std::string text = extract_region_internal(
                data, coords[0], coords[1], coords[2], coords[3]
            );
            if (!text.empty()) {
                results[name] = text;
            }
        }

        return results;
    }
};

PYBIND11_MODULE(fast_ocr, m) {
    py::class_<FastOCR>(m, "FastOCR")
        .def(py::init<int, int>(), py::arg("width") = 3840, py::arg("height") = 2160)
        .def("extract_all_text", &FastOCR::extract_all_text);
}