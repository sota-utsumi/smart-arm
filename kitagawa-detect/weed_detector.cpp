#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;

namespace {

const fs::path kBaseDir = fs::current_path();
const fs::path kSourceDir = kBaseDir / "images" / "source";
const fs::path kOutputDir = kBaseDir / "output";
const fs::path kModelDir = kBaseDir / "model";
const fs::path kModelPath = kModelDir / "weed_classifier.yml";
const fs::path kScalerPath = kModelDir / "weed_scaler.yml";

const cv::Size kImageSize(128, 128);
const cv::Scalar kHsvLower(25, 40, 40);
const cv::Scalar kHsvUpper(95, 255, 255);
constexpr double kMinContourArea = 500.0;

const std::vector<std::string> kImageExtensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
};

struct WeedRegion {
    cv::Rect bbox;
    int areaPx = 0;
    double areaRatio = 0.0;
    cv::Point center;
    std::vector<cv::Point> contour;
};

struct CameraConfig {
    fs::path configPath;
    std::string cameraRaw;
    bool useIndex = false;
    int cameraIndex = 0;
    int cameraWidth = 0;
    int cameraHeight = 0;
};

struct CsvRow {
    std::string timestamp;
    std::string x;
    std::string y;
    int size = 0;
};

class WeedClassifier {
public:
    void load(const fs::path& modelPath, const fs::path& scalerPath) {
        if (!fs::exists(modelPath)) {
            throw std::runtime_error("model file not found: " + modelPath.string());
        }
        if (!fs::exists(scalerPath)) {
            throw std::runtime_error("scaler file not found: " + scalerPath.string());
        }

        svm_ = cv::Algorithm::load<cv::ml::SVM>(modelPath.string());
        if (svm_.empty()) {
            throw std::runtime_error("failed to load model: " + modelPath.string());
        }

        cv::FileStorage fsStorage(scalerPath.string(), cv::FileStorage::READ);
        if (!fsStorage.isOpened()) {
            throw std::runtime_error("failed to open scaler file for reading");
        }
        fsStorage["feature_mean"] >> featureMean_;
        fsStorage["feature_std"] >> featureStd_;

        if (featureMean_.empty() || featureStd_.empty()) {
            throw std::runtime_error("invalid scaler data");
        }
    }

    int predict(const cv::Mat& sample, double* confidence = nullptr) const {
        if (svm_.empty()) {
            throw std::runtime_error("classifier is not loaded");
        }

        cv::Mat normalized = normalize(sample);
        float predicted = svm_->predict(normalized);

        if (confidence != nullptr) {
            cv::Mat rawOutput;
            svm_->predict(normalized, rawOutput, cv::ml::StatModel::RAW_OUTPUT);
            double margin = std::abs(rawOutput.at<float>(0, 0));
            *confidence = 1.0 / (1.0 + std::exp(-margin));
        }

        return static_cast<int>(predicted);
    }

private:
    cv::Mat normalize(const cv::Mat& samples) const {
        cv::Mat repeatedMean;
        cv::repeat(featureMean_, samples.rows, 1, repeatedMean);
        cv::Mat repeatedStd;
        cv::repeat(featureStd_, samples.rows, 1, repeatedStd);

        cv::Mat normalized;
        cv::subtract(samples, repeatedMean, normalized);
        cv::divide(normalized, repeatedStd, normalized);
        return normalized;
    }

    cv::Mat featureMean_;
    cv::Mat featureStd_;
    cv::Ptr<cv::ml::SVM> svm_;
};

std::string toLower(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); }
    );
    return value;
}

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string nowTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);

    std::tm localTm{};
#ifdef _WIN32
    localtime_s(&localTm, &nowTime);
#else
    localtime_r(&nowTime, &localTm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&localTm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

std::vector<fs::path> getImagePaths(const fs::path& directory) {
    std::vector<fs::path> paths;
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return paths;
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string extension = toLower(entry.path().extension().string());
        if (std::find(kImageExtensions.begin(), kImageExtensions.end(), extension) !=
            kImageExtensions.end()) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    return paths;
}

cv::Mat extractFeatures(const cv::Mat& imageBgr) {
    cv::Mat resized;
    cv::resize(imageBgr, resized, kImageSize);

    cv::Mat hsv;
    cv::cvtColor(resized, hsv, cv::COLOR_BGR2HSV);

    std::vector<float> features;
    features.reserve(101);

    for (int channel = 0; channel < 3; ++channel) {
        const int histSize[] = {32};
        const float range[] = {0.0f, 256.0f};
        const float* ranges[] = {range};
        const int channels[] = {channel};
        cv::Mat hist;
        cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
        cv::normalize(hist, hist);

        for (int i = 0; i < hist.rows; ++i) {
            features.push_back(hist.at<float>(i, 0));
        }
    }

    cv::Mat greenMask;
    cv::inRange(hsv, kHsvLower, kHsvUpper, greenMask);
    const double greenRatio =
        static_cast<double>(cv::countNonZero(greenMask)) / static_cast<double>(greenMask.total());
    features.push_back(static_cast<float>(greenRatio));

    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    cv::Mat sobelX;
    cv::Mat sobelY;
    cv::Sobel(gray, sobelX, CV_64F, 1, 0, 3);
    cv::Sobel(gray, sobelY, CV_64F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(sobelX, sobelY, magnitude);

    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(magnitude, mean, stddev);
    double maxValue = 0.0;
    cv::minMaxLoc(magnitude, nullptr, &maxValue);
    cv::Mat thresholdMask = magnitude > mean[0];
    const double activeRatio =
        static_cast<double>(cv::countNonZero(thresholdMask)) / static_cast<double>(thresholdMask.total());

    features.push_back(static_cast<float>(mean[0]));
    features.push_back(static_cast<float>(stddev[0]));
    features.push_back(static_cast<float>(maxValue));
    features.push_back(static_cast<float>(activeRatio));

    cv::Mat featureRow(1, static_cast<int>(features.size()), CV_32F);
    for (int i = 0; i < featureRow.cols; ++i) {
        featureRow.at<float>(0, i) = features[static_cast<size_t>(i)];
    }
    return featureRow;
}

cv::Point contourCenter(const std::vector<cv::Point>& contour, const cv::Rect& bbox) {
    const cv::Moments moments = cv::moments(contour);
    if (std::abs(moments.m00) > 1e-6) {
        return cv::Point(
            static_cast<int>(moments.m10 / moments.m00),
            static_cast<int>(moments.m01 / moments.m00)
        );
    }

    return cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
}

std::vector<WeedRegion> detectWeedRegions(const cv::Mat& imageBgr) {
    cv::Mat hsv;
    cv::cvtColor(imageBgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, kHsvLower, kHsvUpper, mask);

    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double totalArea = static_cast<double>(imageBgr.rows * imageBgr.cols);
    std::vector<WeedRegion> regions;

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < kMinContourArea) {
            continue;
        }

        const cv::Rect bbox = cv::boundingRect(contour);
        WeedRegion region;
        region.bbox = bbox;
        region.areaPx = static_cast<int>(area);
        region.areaRatio = (area / totalArea) * 100.0;
        region.center = contourCenter(contour, bbox);
        region.contour = contour;
        regions.push_back(region);
    }

    std::sort(
        regions.begin(),
        regions.end(),
        [](const WeedRegion& lhs, const WeedRegion& rhs) { return lhs.areaPx > rhs.areaPx; }
    );
    return regions;
}

CameraConfig loadCameraConfig(const fs::path& configPath) {
    if (!fs::exists(configPath)) {
        throw std::runtime_error("config file not found: " + configPath.string());
    }

    std::ifstream input(configPath);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open config file: " + configPath.string());
    }

    std::string cameraRaw;
    std::optional<int> width;
    std::optional<int> height;

    std::string line;
    int lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.front() == '#') {
            continue;
        }

        const auto delimiter = trimmed.find('=');
        if (delimiter == std::string::npos) {
            throw std::runtime_error(
                "invalid config line " + std::to_string(lineNumber) + ": " + trimmed
            );
        }

        const std::string key = trim(trimmed.substr(0, delimiter));
        const std::string value = trim(trimmed.substr(delimiter + 1));
        if (key.empty()) {
            throw std::runtime_error("empty config key at line " + std::to_string(lineNumber));
        }

        if (key == "camera") {
            cameraRaw = value;
        } else if (key == "camera_width") {
            width = std::stoi(value);
        } else if (key == "camera_height") {
            height = std::stoi(value);
        }
    }

    if (cameraRaw.empty() || !width.has_value() || !height.has_value()) {
        throw std::runtime_error("config requires camera, camera_width, camera_height");
    }
    if (*width <= 0 || *height <= 0) {
        throw std::runtime_error("camera_width and camera_height must be positive");
    }

    CameraConfig config;
    config.configPath = configPath;
    config.cameraRaw = cameraRaw;
    config.cameraWidth = *width;
    config.cameraHeight = *height;

    const bool numericSource = !cameraRaw.empty() &&
        std::all_of(cameraRaw.begin(), cameraRaw.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; });

    if (numericSource) {
        config.useIndex = true;
        config.cameraIndex = std::stoi(cameraRaw);
    }

    return config;
}

cv::VideoCapture openCamera(const CameraConfig& config) {
    cv::VideoCapture capture;
    if (config.useIndex) {
        capture.open(config.cameraIndex);
    } else {
        capture.open(config.cameraRaw);
    }

    if (!capture.isOpened()) {
        throw std::runtime_error("failed to open camera: " + config.cameraRaw);
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, config.cameraWidth);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, config.cameraHeight);
    return capture;
}

void writeDetectionCsv(const fs::path& csvPath, const std::vector<CsvRow>& rows) {
    std::ofstream output(csvPath);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open csv for writing: " + csvPath.string());
    }

    output << "timestamp,x,y,size\n";
    for (const auto& row : rows) {
        output << row.timestamp << "," << row.x << "," << row.y << "," << row.size << "\n";
    }
}

void processSourceImages() {
    std::cout << "============================================================\n";
    std::cout << "Processing source images with C++ detector\n";
    std::cout << "============================================================\n";

    const auto sourceFiles = getImagePaths(kSourceDir);
    if (sourceFiles.empty()) {
        throw std::runtime_error("no source images found in " + kSourceDir.string());
    }

    WeedClassifier classifier;
    classifier.load(kModelPath, kScalerPath);

    fs::create_directories(kOutputDir);
    const fs::path csvPath = kOutputDir / "detections_cpp.csv";

    std::vector<CsvRow> rows;
    rows.reserve(sourceFiles.size());

    for (size_t index = 0; index < sourceFiles.size(); ++index) {
        const auto& path = sourceFiles[index];
        std::cout << "[" << (index + 1) << "/" << sourceFiles.size() << "] Processing "
                  << path.filename().string() << " ...\n";

        const cv::Mat image = cv::imread(path.string());
        if (image.empty()) {
            std::cout << "  -> failed to read image\n";
            continue;
        }

        double confidence = 0.0;
        const int predicted = classifier.predict(extractFeatures(image), &confidence);
        const std::string label = predicted == 1 ? "weed" : "not_weed";
        std::cout << "  Classification: " << label
                  << " (confidence: " << std::fixed << std::setprecision(2)
                  << confidence * 100.0 << "%)\n";

        const auto regions = detectWeedRegions(image);
        std::cout << "  Weed regions: " << regions.size() << "\n";

        CsvRow row;
        row.timestamp = nowTimestamp();

        if (!regions.empty()) {
            const auto& largest = regions.front();
            row.x = std::to_string(largest.center.x);
            row.y = std::to_string(largest.center.y);
            row.size = largest.areaPx;

            std::cout << "  Largest weed center: (" << largest.center.x
                      << ", " << largest.center.y << ") px\n";
        } else {
            row.size = 0;
        }
        rows.push_back(row);

        for (size_t regionIndex = 0; regionIndex < regions.size(); ++regionIndex) {
            const auto& region = regions[regionIndex];
            std::cout << "    #" << (regionIndex + 1)
                      << ": center=(" << region.center.x << ", " << region.center.y << ") px, "
                      << "bbox=(" << region.bbox.x << ", " << region.bbox.y << ", "
                      << region.bbox.width << ", " << region.bbox.height << "), "
                      << "area=" << region.areaPx << "px ("
                      << std::fixed << std::setprecision(2) << region.areaRatio << "%)\n";
        }

        std::cout << "\n";
    }

    writeDetectionCsv(csvPath, rows);

    std::cout << "============================================================\n";
    std::cout << "Done. Detection CSV was saved to: " << csvPath << "\n";
    std::cout << "============================================================\n";
}

void processCameraStream(const fs::path& configPath) {
    const CameraConfig cameraConfig = loadCameraConfig(configPath);

    std::cout << "============================================================\n";
    std::cout << "Starting live camera detection (C++)\n";
    std::cout << "============================================================\n";
    std::cout << "Config: " << cameraConfig.configPath << "\n";
    std::cout << "Camera: " << cameraConfig.cameraRaw << "\n";
    std::cout << "Requested size: " << cameraConfig.cameraWidth
              << "x" << cameraConfig.cameraHeight << "\n";

    cv::VideoCapture capture = openCamera(cameraConfig);
    const int actualWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int actualHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (actualWidth > 0 && actualHeight > 0) {
        std::cout << "Opened size: " << actualWidth << "x" << actualHeight << "\n";
    }

    std::cout << "Press Ctrl+C to stop.\n";

    while (true) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            throw std::runtime_error("failed to read a frame from the camera");
        }

        const auto regions = detectWeedRegions(frame);
        if (regions.empty()) {
            std::cout << "\rLargest weed center: not detected      " << std::flush;
            continue;
        }

        const auto& largest = regions.front();
        std::cout << "\rLargest weed center: (" << largest.center.x << ", " << largest.center.y
                  << ") px | bbox=(" << largest.bbox.x << ", " << largest.bbox.y << ", "
                  << largest.bbox.width << ", " << largest.bbox.height << ") | area="
                  << largest.areaPx << "px      " << std::flush;
    }
}

struct Options {
    bool detect = false;
    bool cameraDetect = false;
    std::optional<fs::path> configPath;
};

void printUsage(const char* programName) {
    std::cout << "Usage:\n";
    std::cout << "  " << programName << " --detect\n";
    std::cout << "  " << programName << " --camera-detect --config config.txt\n";
}

Options parseArgs(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--detect") {
            options.detect = true;
        } else if (arg == "--camera-detect") {
            options.cameraDetect = true;
        } else if (arg == "--config") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--config requires a path");
            }
            options.configPath = fs::path(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);

        if (!options.detect && !options.cameraDetect) {
            printUsage(argv[0]);
            return 0;
        }

        if (options.detect) {
            processSourceImages();
        }

        if (options.cameraDetect) {
            if (!options.configPath.has_value()) {
                throw std::runtime_error("--camera-detect requires --config PATH");
            }
            processCameraStream(*options.configPath);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }
}
