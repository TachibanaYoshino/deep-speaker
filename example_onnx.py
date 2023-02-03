import random
import tensorflow as tf
import numpy as np
import tf2onnx
import onnxruntime as rt

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

# Sample some inputs for WAV/FLAC files for the same speaker.
# To have reproducible results every time you call this function, set the seed every time before calling it.
# np.random.seed(123)
# random.seed(123)
mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
mfcc_002 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)

# Call the model to get the embeddings of shape (1, 512) for each file.
predict_001 = model.m.predict(np.expand_dims(mfcc_001, axis=0))
predict_002 = model.m.predict(np.expand_dims(mfcc_002, axis=0))

# Do it again with a different speaker.
mfcc_003 = sample_from_mfcc(read_mfcc('samples/1255-90413-0001.flac', SAMPLE_RATE), NUM_FRAMES)
predict_003 = model.m.predict(np.expand_dims(mfcc_003, axis=0))

# Compute the cosine similarity and check that it is higher for the same speaker.
same_speaker_similarity = batch_cosine_similarity(predict_001, predict_002)
diff_speaker_similarity = batch_cosine_similarity(predict_001, predict_003)
print('SAME SPEAKER', same_speaker_similarity)  # SAME SPEAKER [0.81564593]
print('DIFF SPEAKER', diff_speaker_similarity)  # DIFF SPEAKER [0.1419204]

assert same_speaker_similarity > diff_speaker_similarity

# tf.saved_model.save(model,"./checkpoints/pb")
print('Start')
# 定义模型转onnx的参数
spec = (tf.TensorSpec((1, 160, 64, 1), tf.float32, name="input"),)  # 输入签名参数，(None, 128, 128, 3)决定输入的size
output_path = "deep_speaker.onnx"
# 转换并保存onnx模型，opset决定选用的算子集合
model_proto, _ = tf2onnx.convert.from_keras(model.m, input_signature=spec, opset=12, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)  # 查看输出名称，后面推理用的到

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
# 推理onnx模型
output_names = output_names
onnx_pred1 = m.run(None, {"input": np.expand_dims(mfcc_001, axis=0)})[0]
onnx_pred2 = m.run(None, {"input": np.expand_dims(mfcc_002, axis=0)})[0]
print(batch_cosine_similarity(predict_001, predict_002))

