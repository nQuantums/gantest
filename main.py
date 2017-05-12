"""メインとは名ばかりのテスト用ソース.
"""
import numpy as np
import dnn
import dsconf
import selector

def main():
	dnn.Init(0)

	# 指定されたモデルのルート取得
	mr = selector.SelectByName("CycleGan")

	# 学習モデル作成
	m = mr.CreateModel("CycleGan", dsconf.InChs, dsconf.OutChs)

	# 学習済みデータがあれば読み込む
	print("Loading trained data...")
	m.load()
	print("Done.")

	# 学習結果を保存
	print("Saving trained data...")
	m.save()
	print("Done.")


if __name__ == "__main__":
	main()
