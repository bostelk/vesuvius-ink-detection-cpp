#include <stdio.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>

#include "image.h"

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

static float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

template <typename T>
static void sigmoid(T& input) {
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = sigmoid(input[i]);
    }
}


// This is the structure to interface with the ink detection model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct YoussefInkDetection {
    YoussefInkDetection() {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
            input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
            output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run() {
        const char* input_names[] = { "arg0" };
        const char* output_names[] = { "decoder_1" };

        Ort::RunOptions run_options;
        //Ort::GetApi().RunOptionsSetRunLogSeverityLevel(run_options, 0); VERBOSE
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        sigmoid(results_);
        // There is no single "most confident" label to return.
        return result_;
    }

    static constexpr const int width_ = 32;
    static constexpr const int height_ = 32;

    std::array<float, width_* height_> input_image_{};
    std::array<float, 8 * 8> results_{};
    int64_t result_{ 0 };

private:
    Ort::Env env;
    Ort::Session session_{ env, L"vesuvius-ink-detection.onnx", Ort::SessionOptions{nullptr} };

    Ort::Value input_tensor_{ nullptr };
    std::array<int64_t, 4> input_shape_{ 1, 1, width_, height_ };

    Ort::Value output_tensor_{ nullptr };
    std::array<int64_t, 4> output_shape_{ 1, 1, 8, 8 };
};

std::unique_ptr<YoussefInkDetection> inkDetection_;

void ZeroInput()
{
    std::fill(inkDetection_->input_image_.begin(), inkDetection_->input_image_.end(), 0.f);
}

int main()
{
	printf("hello world");
    try {
        inkDetection_ = std::make_unique<YoussefInkDetection>();
    }
    catch (const Ort::Exception& exception) {
        //MessageBoxA(nullptr, exception.what(), "Error:", MB_OK);
        return 0;
    }

    // https://vesuvius.virtual-void.net/scroll/1/segment/20230827161847/#u=2524&v=4582&zoom=0.046&rot=90&flip=false&layer=34

    int x0 = 1248;
    int y0 = 9163 / 2;
    int layerImageWidth = 5048;
    int layerImageHeight = 9163;

    int pixelStride = 2; 

    Image image(layerImageWidth, layerImageHeight); // Allocates.

    for (int y = y0; y < image.height; y+= pixelStride)
    {
        for (int x = x0; x < image.width; x+= pixelStride)
        {
            printf((std::to_string(x) + " " + std::to_string(y) + "\n").c_str());

            // Todo(kbostelmmann): Erode layer mask by a 16x16 kernel.
            auto mask = ReadImage("20230827161847/20230827161847_mask.png");
            if (!mask || mask->GetPixelGray(x, y) != 255)
            {
                continue; // Skip because pixel is not on papyrus.
            }

            // Todo(kbostelmann): Read layer x until y (+30 slices). Fill into inputs.
            int initialLayerIndex = 15;
            int layerIndexDistance = 1;
            for (int layerIndex = initialLayerIndex; layerIndex < initialLayerIndex + layerIndexDistance; layerIndex++)
            {
                // Todo(kbostelmann): Pad image to nearest tile boundary. Fill new pixels with zero.
                auto layer = ReadImage("20230827161847/layers/" + std::to_string(layerIndex) + ".tif");
                if (layer) {
                    ZeroInput();
                    CopyToInput(*layer, x, y, inkDetection_->input_image_.data(), YoussefInkDetection::width_, YoussefInkDetection::height_);
                }
            }

            inkDetection_->Run();

            auto imageTile = ConvertResultToImage(inkDetection_->results_.data());

            // Resize to input resolution.
            auto imageTileResized = ResizeImage(imageTile, YoussefInkDetection::width_, YoussefInkDetection::height_);

            // Todo(kbostelmann): Accumulate predictions and then normalize.
            CopyToImage(*imageTileResized, image, x, y);
            std::string filename = "out.png";
            WriteImage(filename, image);
        }
    }
}