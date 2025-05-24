#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <map>
#include <vector>

namespace py = pybind11;

class FastOCR {
private:
    // Define HOI4 UI regions (for 1920x1080)
    std::map<std::string, std::vector<int>> regions;

public:
    FastOCR() {
        // Initialize regions - x, y, width, height
        regions["date"] = {860, 10, 200, 40};
        regions["political_power"] = {250, 10, 100, 40};
        regions["factories"] = {1700, 10, 200, 50};
        regions["manpower"] = {400, 10, 150, 40};
        regions["division_count"] = {1700, 900, 200, 50};
    }

    std::map<std::string, std::string> extract_all_text(py::array_t<uint8_t> image) {
        std::map<std::string, std::string> results;

        // Get image dimensions
        auto buf = image.request();
        if (buf.ndim != 3) {
            throw std::runtime_error("Image must be 3D (height, width, channels)");
        }

        int height = buf.shape[0];
        int width = buf.shape[1];

        // For now, return dummy data based on regions
        // We'll add real OCR after confirming this compiles
        results["date"] = "January 1936";
        results["political_power"] = "150";
        results["factories"] = "42 17";
        results["manpower"] = "2.18M";
        results["division_count"] = "31";

        // In the real version, we'll process each region with Tesseract
        return results;
    }

    std::string extract_region(py::array_t<uint8_t> image, const std::string& region_name) {
        if (regions.find(region_name) == regions.end()) {
            return "";
        }

        // For now, return dummy data
        if (region_name == "date") return "January 1936";
        if (region_name == "political_power") return "150";
        if (region_name == "factories") return "42 17";

        return "";
    }
};

PYBIND11_MODULE(fast_ocr, m) {
    py::class_<FastOCR>(m, "FastOCR")
        .def(py::init<>())
        .def("extract_all_text", &FastOCR::extract_all_text)
        .def("extract_region", &FastOCR::extract_region);
}