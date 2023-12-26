#include <array>

// A square (width = height) subset of an image.
template <typename T, int U, int V>
struct ImageTile
{
    static constexpr const int width = U; // pixels
    static constexpr const int height = U; // pixels
    static constexpr const int channels = V; // intensity
    size_t strideBytes = channels * width * sizeof(T);
    int pixelCount = width * height;
    std::array<T, width* height* channels> pixels{}; // color data
};

// A window of a slice to detect ink within.
typedef ImageTile<float, 64, 1> InputImage;

// An ink detection result converted to an image.
typedef ImageTile<uint8_t, 16, 3> ResultImage;

// A generic image. Allocates heap memory on construction.
struct Image
{
    int width; // pixels
    int height; // pixels
    int channels; // rgb
    size_t strideBytes() {
        return channels * width * sizeof(uint8_t);
    }
    size_t sizeBytes() {
        return height * strideBytes();
    }
    int pixelCount() {
        return width * height;
    }
    uint8_t* pixels; // color data
    uint8_t GetPixelGray(int x, int y) {
        int pixelOffset = y * width + x;
        return pixels[pixelOffset * channels + 0];
    }
    Image(int width_, int height_) : width(width_), height(height_), channels(3) {
        pixels = (uint8_t*)malloc(sizeBytes());
        memset(pixels, 0, sizeBytes());
    }
    Image(int width_, int height_, int channels_) : width(width_), height(height_), channels(channels_) {
        pixels = (uint8_t*)malloc(sizeBytes());
        memset(pixels, 0, sizeBytes());
    }
    ~Image() {
        free(pixels);
    }
};

std::unique_ptr<Image> ReadImageTIFF(std::string filename);
void CopyToInput(Image& image, int x0, int y0, float* input, int inputWidth, int inputHeight);
ResultImage ConvertResultToImage(float* results);
std::unique_ptr<Image> ReadImage(std::string filename);
void WriteImagePNG(std::string filename, Image& image);
void WriteImage(std::string filename, Image& image);
// Copy a source image into a destination image region located at (x0, y0).
void CopyToImage(Image& src, Image& dst, int x0, int y0);
void FillPixels(ResultImage& image, uint8_t red, uint8_t blue, uint8_t green);
std::unique_ptr<Image> ResizeImage(ResultImage& image, int width, int height);