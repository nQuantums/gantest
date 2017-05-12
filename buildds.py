"""データセット作成処理.
"""
import argparse
import selector
import dsconf

def main():
	# コマンドライン引数解析
	parser = argparse.ArgumentParser(description='Build dataset for specified model.')
	parser.add_argument("modelName", help='Model name. Currently CycleGAN only.')
	parser.add_argument("sourceDir", help='Source images directory.')
	args = parser.parse_args()

	# 指定モデルのモジュール取得
	mr = selector.SelectByName(args.modelName)

	# データセットビルダ取得しデータセット作成
	b = mr.GetDsBuilder()
	b.Build(args.sourceDir, dsconf.GetDir())


if __name__ == "__main__":
	main()
