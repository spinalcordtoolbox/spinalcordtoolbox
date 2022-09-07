# Functions dealing with onnxruntime (used by sct_deepseg_sc, sct_deepseg_gm, and sct_deepseg_lesion)

import onnxruntime as ort


def onnx_inference(model_path, input_data):
    """Perform inference using an '.onnx' model."""
    # This option helps to combat an issue where the CPU memory arena would unnecessarily
    # consume ~6 gigabytes of memory, when our models only truly use ~5 megabytes. See also:
    # https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3738#discussion_r881735426
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False

    ort_sess = ort.InferenceSession(model_path, sess_options=sess_options)
    preds = ort_sess.run(output_names=["predictions"], input_feed={"input_1": input_data})

    return preds
