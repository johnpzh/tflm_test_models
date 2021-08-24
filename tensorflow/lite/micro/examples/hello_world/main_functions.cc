/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/hello_world/main_functions.h"
// #include <sys/types.h>
#include <iomanip>

// #include "stm32469i_discovery.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/constants.h"
// #include "tensorflow/lite/micro/examples/hello_world/model.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_MobileNetV1.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mnist.h"
// #include "tensorflow/lite/micro/examples/hello_world/mbv2-w0.3-r80_imagenet.h"
// #include "tensorflow/lite/micro/examples/hello_world/proxyless-w0.25-r112_imagenet.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_79M.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_103M.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_124M.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_176M_66top1.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_79M_quan_int8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_103M_quan_int8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_124M_quan_int8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_176M_66top1_quan_int8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_79M_dynamic_quan_int8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_42M.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_42M_quan_int8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_42M_dynamic_quan_int8.h"
#include "tensorflow/lite/micro/examples/hello_world/model_mcunet_42M_uint8.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_mcunet_42M_fp32.h"
// #include "tensorflow/lite/micro/examples/hello_world/model_food.h"
#include "tensorflow/lite/micro/examples/hello_world/output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "misc.h"
#include "common.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// constexpr int kTensorArenaSize = 1000; // for Food Model
constexpr int kTensorArenaSize = 305 * 1024; // for MobileNetV2 in MCUNet
// constexpr int kTensorArenaSize = 306 * 1024; // for MobileNetV2 in MCUNet
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
int setup() {
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  //   model = tflite::GetModel(g_model);
  //   model = tflite::GetModel(mobile_net_v1);
//   model = tflite::GetModel(mbv2_w0_3_r80_imagenet_tflite); // MobileNetV2 from MCUNet
//   model = tflite::GetModel(proxyless_w0_25_r112_imagenet_tflite);
//   model = tflite::GetModel(mcunet_79M_tflite);
//   model = tflite::GetModel(mcunet_103M_tflite);
//   model = tflite::GetModel(mcunet_124M_tflite);
//   model = tflite::GetModel(mcunet_176M_66top1_tflite);
//   model = tflite::GetModel(mcunet_79M_quan_int8_tflite);
//   model = tflite::GetModel(mcunet_103M_quan_int8_tflite);
//   model = tflite::GetModel(mcunet_124M_quan_int8_tflite);
//   model = tflite::GetModel(mcunet_176M_66top1_quan_int8_tflite);
//   model = tflite::GetModel(mcunet_79M_dynamic_quan_int8_tflite);
//   model = tflite::GetModel(mcunet_42M_tflite);
//   model = tflite::GetModel(mcunet_42M_quan_int8_tflite);
//   model = tflite::GetModel(mcunet_42M_dynamic_quan_int8_tflite);
  model = tflite::GetModel(mcunet_42M_uint8_tflite);
//   model = tflite::GetModel(mcunet_42M_fp32_tflite);
//   model = tflite::GetModel(model_food_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return -1;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;

  //   {//
  //     TF_LITE_REPORT_ERROR(error_reporter, "input->dims->size: %d\n",
  //     input->dims->size);
  //   }
  return 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
    float run_time = -get_time_mark();
    for (uint32_t loop_i = 0; loop_i < LOOP_SIZE; ++loop_i) { 
        for (uint32_t i = 0; i < INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * INPUT_IMAGE_CHANNEL; ++i) {
            input->data.int8[i] = 0;
        // for (uint32_t i = 0; i < 65; ++i) { // for Food Model
            // input->data.f[i] = 0; // float32?
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.\n");
            return;
        }

        // for (uint32_t i = 0; i < OUTPUT_VECTOR_SIZE; ++i) {
        //     TF_LITE_REPORT_ERROR(error_reporter, "[i]: %d ", output->data.int8[i]);
        // }
        // TF_LITE_REPORT_ERROR(error_reporter, "\n");
    }
    run_time += get_time_mark();
    TF_LITE_REPORT_ERROR(error_reporter, 
                            "run_time(s.): %f "
                            "latency(ms.): %f ",
                            run_time,
                            run_time / LOOP_SIZE * 1000.0);

//   {  //
//      // for (int i = 0; i < 100; ++i) {
//     TF_LITE_REPORT_ERROR(error_reporter,
//                          "input->dims->size: %d "
//                          "input->bytes: %d\n",
//                          input->dims->size, input->bytes);
//     // }
//     wait_ms(1000);
//   }
}
