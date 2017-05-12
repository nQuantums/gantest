import os
from pathlib import Path
import numpy as np
import cv2
import dsconf
import imgutil

def Build(imagesDir, datasetsDir):
	"""指定ディレクトリ内の画像ファイルからデータセットを作成する.
	# Args:
		imagesDir: 画像ファイルが入ったディレクトリ名.
		datasetsDir: 出力先データセットディレクトリ名.
	"""

	# CycleGAN用データセットディレクトリ作成
	cgd = os.path.join(datasetsDir, "cyclegan")
	if not os.path.isdir(cgd):
		os.mkdir(cgd)

	# 出力先データセットディレクトリに白バックとそれ以外用ディレクトリ作成
	nbd = os.path.join(cgd, "nb")
	if not os.path.isdir(nbd):
		os.mkdir(nbd)
	wbd = os.path.join(cgd, "wb")
	if not os.path.isdir(wbd):
		os.mkdir(wbd)

	maxSize = max(dsconf.MaxImageSize[0], dsconf.MaxImageSize[1])

	# 指定ディレクトリ内の画像ファイル一覧取得
	p = Path(imagesDir)
	pls = []
	pls.extend(p.glob("*.jpg"))
	pls.extend(p.glob("*.png"))
	pls.extend(p.glob("*.jpeg"))
	imageCount = len(pls)
	convertedCount = 0

	for pl in pls:
		file = os.path.normpath(os.path.join(imagesDir, pl.name))
		img = cv2.imread(file, cv2.IMREAD_COLOR)
		if img is None or img.shape[0] < 128 or img.shape[1] < 128:
			imageCount -= 1
			continue

		# モノトーンはスキップする
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
		hsv = hsv.transpose(2, 0, 1)
		if np.mean(hsv[1]) < 20:
			imageCount -= 1
			continue

		# イメージを所定のサイズへ縮小
		img = imgutil.ResizeIfLarger(img, maxSize)

		# 白バックかどうか判定して保存先ディレクトリを分けて保存
		d = wbd if _isWhiteBack(img) else nbd
		f = os.path.splitext(pl.name)[0]
		cv2.imwrite(os.path.normpath(os.path.join(d, f + ".png")), img)

		convertedCount += 1
		print(convertedCount, "/", imageCount)

def _isWhiteBack(img):
	"""指定イメージが白バックかどうか判定する
	# Args:
		img: 判定元イメージ
	# Returns:
		白バックなら True.
	"""
	img = img.transpose(2, 0, 1)

	# イメージの淵を取得し、白かどうか判定する
	t = img[:, :16, :]
	b = img[:, -16:, :]
	l = img[:, :, :16]
	r = img[:, :, -16:]

	# 淵の最も暗い部分が指定値より明るければ白バックとする
	threshold = 200
	if threshold <= np.min(t) and threshold <= np.min(b) and threshold <= np.min(l) and threshold <= np.min(r):
		return True

	# 淵の平均値が指定値より明るくても白バックとする
	threshold = 230
	if threshold <= np.mean(t) and threshold <= np.mean(b) and threshold <= np.mean(l) and threshold <= np.mean(r):
		return True

	return False
