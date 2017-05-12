"""データセットロード処理のテスト.
"""
import cv2
import selector
import dsconf

def main():
	m = selector.SelectByName("CycleGan")
	l = m.GetDsLoader()
	cnv = m.GetImgCnv()
	ds = l.Load(dsconf.GetDir())
	index = 0
	keypressed = True
	while True:
		# キーが押されたら
		if keypressed:
			keypressed = False
			x, t = ds[index].get()
			cv2.imshow('x', cnv.DnnToBgr(x))
			cv2.imshow('t', cnv.DnnToBgr(t))

		k = cv2.waitKey(0)
		if k == 27:
			break
		elif k == 52:
			keypressed = True
			index -= 1
			if index < 0:
				index = 0
		elif k == 54:
			keypressed = True
			index += 1
			if len(ds) <= index:
				index = len(ds) - 1


if __name__ == "__main__":
	main()
