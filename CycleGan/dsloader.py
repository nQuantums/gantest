import os
from pathlib import Path
import cv2
import random
import numpy as np
import dnn
import dsconf
import imgutil
from . import imgcnv

class Dataset():
	"""データセット.
	"""
	def __init__(self, hash, xs, ts, xrev, yrev):
		self.hash = hash
		self.xs = xs
		self.ts = ts
		self.xrev = xrev
		self.yrev = yrev

	def get(self):
		"""データセットを取得する.
		# Returns:
			(入力データ, 教師データ)
		"""
		dssize = dsconf.MaxImageSize
		maxSize = dssize[0]
		halfSize = maxSize // 2
		size1 = random.randint(halfSize, maxSize)
		size2 = random.randint(halfSize, maxSize)

		# リサイズ後の画像を作成
		rx = imgutil.ResizeIfLarger(self.xs[random.randrange(0, len(self.xs))], size1)
		rt = imgutil.ResizeIfLarger(self.ts[random.randrange(0, len(self.ts))], size2)
		rx = imgcnv.BgrToDnn(rx)
		rt = imgcnv.BgrToDnn(rt)

		# データペア領域作成
		x = np.full((dsconf.InChs,) + dssize, 1, dtype=dnn.dtype)
		t = np.full((dsconf.OutChs,) + dssize, 1, dtype=dnn.dtype)

		# リサイズ後の画像をランダムな位置に配置
		w = rx.shape[2]
		h = rx.shape[1]
		ox = random.randint(0, dssize[1] - w)
		oy = random.randint(0, dssize[0] - h)
		x[:, oy:oy + h, ox:ox + w] = rx[:, ::self.yrev, ::self.xrev]

		w = rt.shape[2]
		h = rt.shape[1]
		ox = random.randint(0, dssize[1] - w)
		oy = random.randint(0, dssize[0] - h)
		t[:, oy:oy + h, ox:ox + w] = rt[:, ::self.yrev, ::self.xrev]

		return x, t

def Load(datasetsDir):
	"""データセットディレクトリから全データセットを読み込む.
	# Args:
		datasetsDir: データセットディレクトリ名.
	# Returns:
		データセットのリスト.
	"""
	# CycleGAN用データセットディレクトリ
	cgd = os.path.join(datasetsDir, "cyclegan")
	# 白バックとそれ以外用ディレクトリ名
	nbd = os.path.join(cgd, "nb")
	wbd = os.path.join(cgd, "wb")

	# 白バック以外のイメージ読み込み
	p = Path(nbd)
	pls = []
	pls.extend(p.glob("*.jpg"))
	pls.extend(p.glob("*.png"))
	pls.extend(p.glob("*.jpeg"))
	imageCount = len(pls)
	loadedCount = 0
	xs = []
	for pl in pls:
		file = os.path.normpath(os.path.join(nbd, pl.name))
		img = cv2.imread(file, cv2.IMREAD_COLOR)
		if img is None:
			imageCount -= 1
			continue
		xs.append(img)
		loadedCount += 1
		print(loadedCount, "/", imageCount)

	# 白バックのイメージ読み込み
	p = Path(wbd)
	pls = []
	pls.extend(p.glob("*.jpg"))
	pls.extend(p.glob("*.png"))
	pls.extend(p.glob("*.jpeg"))
	imageCount = len(pls)
	loadedCount = 0
	ts = []
	for pl in pls:
		file = os.path.normpath(os.path.join(wbd, pl.name))
		img = cv2.imread(file, cv2.IMREAD_COLOR)
		if img is None:
			imageCount -= 1
			continue
		ts.append(img)
		loadedCount += 1
		print(loadedCount, "/", imageCount)

	# データセットリストに追加
	ds = []
	for i in range(len(xs) + len(ts)):
		ds.append(Dataset(i, xs, ts, 1, 1))
		ds.append(Dataset(i, xs, ts, 1, -1))
		ds.append(Dataset(i, xs, ts, -1, 1))
		ds.append(Dataset(i, xs, ts, -1, -1))
	random.shuffle(ds)
	return ds
