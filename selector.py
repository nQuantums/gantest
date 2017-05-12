"""CycleGAN／pix2pixなどモジュール選択処理.
"""

def SelectByName(name):
	"""指定されたモデル名からネットワークモデルやらデータセットローダーやら含んだモジュール取得.
	# Args:
		name: モデル名.
	# Returns:
		モジュール.
	"""
	if name == "pix2pix":
		import pix2pix
		return pix2pix
	elif name == "CycleGAN":
		import CycleGan
		return CycleGan
	else:
		print(name, "モデルは存在しません。")
		raise
