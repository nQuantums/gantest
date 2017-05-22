"""共通のイメージ処理.
"""
import math
import numpy as np
import cv2
import dnn

def ResizeIfLarger(img, size):
	"""指定イメージが指定サイズを超えているなら縮小する、アスペクト比は維持される.
	# Args:
		img: 元イメージ.
		size: 目標サイズ.
	# Returns:
		イメージ.
	"""
	# リサイズ後の画像を作成
	shape = img.shape
	r = size / max(shape[0], shape[1])
	if 1.0 <= r:
		return img

	w = max(int(math.ceil(shape[1] * r)), 1)
	h = max(int(math.ceil(shape[0] * r)), 1)
	return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def BgrToPM(bgr):
	"""BGRイメージを(3, h, w)形状且つレンジ-1...+1に変換する.
	"""
	pm = bgr.transpose(2, 0, 1).astype(dnn.dtype)
	pm -= 127.5
	pm /= 127.5
	return pm

def PMToBgr(pm):
	"""(3, h, w)形状レンジ-1...+1をBGRイメージに変換する.
	"""
	pm = pm * 127.5
	pm += 127.5
	np.clip(pm, 0, 255, pm)
	bgr = pm.astype(np.uint8)
	bgr = bgr.transpose(1, 2, 0)
	return bgr
