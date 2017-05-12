"""学習実行処理.
"""
import argparse
import numpy as np
import cv2
import dnn
import dsconf
import selector

def main():
	# コマンドライン引数解析
	parser = argparse.ArgumentParser(description='Training of specified model.')
	parser.add_argument("modelName", help='Model name. Currently CycleGAN only.')
	parser.add_argument("--epoch", "-e", type=int, default=10, help='Number of epochs.')
	parser.add_argument("--batch_size", "-b", type=int, default=2, help='Batch size.')
	parser.add_argument("--gpu", "-g", type=int, default=0, help='GPU device number. Use numpy if specified -1.')
	args = parser.parse_args()
	batch_size = args.batch_size

	# GPUなどの環境初期化
	dnn.Init(args.gpu)

	# 指定されたモデルのルート取得
	mr = selector.SelectByName(args.modelName)

	# 学習用データセット読み込み
	print("Loading datasets...")
	ds = mr.GetDsLoader().Load(dsconf.GetDir())
	print("Done.")

	# 学習モデル作成
	m = mr.CreateModel(args.modelName, dsconf.InChs, dsconf.OutChs, batch_size)

	# 学習済みデータがあれば読み込む
	print("Loading trained data...")
	m.load()
	print("Done.")

	# バッチサイズ分のCPU上でのメモリ領域、一旦ここに展開してから toGpu すると無駄が少ない
	xcpu = np.zeros((batch_size, dsconf.InChs, dsconf.MaxImageSize[0], dsconf.MaxImageSize[1]), dtype=dnn.dtype)
	tcpu = np.zeros((batch_size, dsconf.OutChs, dsconf.MaxImageSize[0], dsconf.MaxImageSize[1]), dtype=dnn.dtype)

	# 学習ループ
	requestQuit = False
	iterCount = 0

	for epoch in range(args.epoch):
		if requestQuit:
			break

		print("Epoch", epoch)
		for b in range(len(ds) // batch_size):
			# バッチセットアップ
			for i in range(batch_size):
				index = b * batch_size + i
				d = ds[index]
				xcpu[i, :, :, :], tcpu[i, :, :, :] = d.get()

			# 学習実行
			m.train(dnn.ToGpu(xcpu), dnn.ToGpu(tcpu), iterCount % 10 == 0, iterCount % 10 == 0)
			iterCount += 1

			# OpenCVウィンドウアクティブにしてESCキーで中断できるようにしておく
			k = cv2.waitKey(1)
			if k == 27:
				requestQuit = True
				break

	# 学習結果を保存
	print("Saving trained data...")
	m.save()
	print("Done.")


if __name__ == "__main__":
	main()
