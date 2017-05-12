import os
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import cv2
import dnn
import dsconf
from . import imgcnv
import uploader

# 参考元 https://github.com/Aixile/chainer-cyclegan

def cal_l2_sum(h, t):
	return F.sum((h - t) ** 2) / np.prod(h.data.shape)

def loss_func_rec_l1(x_out, t):
	return F.mean_absolute_error(x_out, t)

def loss_func_rec_l2(x_out, t):
	return F.mean_squared_error(x_out, t)

def loss_func_adv_dis_fake(y_fake):
	return cal_l2_sum(y_fake, 0.1)

def loss_func_adv_dis_real(y_real):
	return cal_l2_sum(y_real, 0.9)

def loss_func_adv_gen(y_fake):
	return cal_l2_sum(y_fake, 0.9)

def add_noise(h, test, sigma=0.2):
	if test:
		return h
	else:
		return h + sigma * dnn.xp.random.randn(*h.data.shape)

class ResBlock(chainer.Chain):
	def __init__(self, ch, bn=True, activation=F.relu):
		self.bn = bn
		self.activation = activation
		layers = {}
		layers['c0'] = L.Convolution2D(ch, ch, 3, 1, 1)
		layers['c1'] = L.Convolution2D(ch, ch, 3, 1, 1)
		if bn:
			layers['bn0'] = L.BatchNormalization(ch)
			layers['bn1'] = L.BatchNormalization(ch)
		super(ResBlock, self).__init__(**layers)

	def __call__(self, x):
		h = self.c0(x)
		if self.bn:
			h = self.bn0(h, test=dnn.test)
		h = self.activation(h)
		h = self.c1(x)
		if self.bn:
			h = self.bn1(h, test=dnn.test)
		return h + x

class CBR(chainer.Chain):
	def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False, noise=False):
		self.bn = bn
		self.activation = activation
		self.dropout = dropout
		self.sample = sample
		self.noise = noise
		layers = {}
		w = chainer.initializers.Normal(0.02)
		if sample == 'down':
			layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
		elif sample == 'none-9':
			layers['c'] = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
		elif sample == 'none-7':
			layers['c'] = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
		elif sample == 'none-5':
			layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
		else:
			layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
		if bn:
			if self.noise:
				layers['batchnorm'] = L.BatchNormalization(ch1, use_gamma=False)
			else:
				layers['batchnorm'] = L.BatchNormalization(ch1)
		super(CBR, self).__init__(**layers)

	def __call__(self, x):
		if self.sample == "down" or self.sample == "none" or self.sample == 'none-9' or self.sample == 'none-7' or self.sample == 'none-5':
			h = self.c(x)
		elif self.sample == "up":
			h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
			h = self.c(h)
		else:
			print("unknown sample method %s" % self.sample)
		if self.bn:
			h = self.batchnorm(h, test=dnn.test)
		if self.noise:
			h = add_noise(h, test=dnn.test)
		if self.dropout:
			h = F.dropout(h, train=not dnn.test)
		if not (self.activation is None):
			h = self.activation(h)
		return h

class Generator_ResBlock_6(chainer.Chain):
	def __init__(self):
		layers = {}
		layers["c1"] = CBR(3, 32, bn=True, sample='none-7')
		layers["c2"] = CBR(32, 64, bn=True, sample='down')
		layers["c3"] = CBR(64, 128, bn=True, sample='down')
		layers["c4"] = ResBlock(128, bn=True)
		layers["c5"] = ResBlock(128, bn=True)
		layers["c6"] = ResBlock(128, bn=True)
		layers["c7"] = ResBlock(128, bn=True)
		layers["c8"] = ResBlock(128, bn=True)
		layers["c9"] = ResBlock(128, bn=True)
		layers["c10"] = CBR(128, 64, bn=True, sample='up')
		layers["c11"] = CBR(64, 32, bn=True, sample='up')
		layers["c12"] = CBR(32, 3, bn=True, sample='none-7', activation=F.tanh)
		super().__init__(**layers)

	def __call__(self, x):
		h = self.c1(x)
		h = self.c2(h)
		h = self.c3(h)
		h = self.c4(h)
		h = self.c5(h)
		h = self.c6(h)
		h = self.c7(h)
		h = self.c8(h)
		h = self.c9(h)
		h = self.c10(h)
		h = self.c11(h)
		h = self.c12(h)
		return h

class Generator_ResBlock_9(chainer.Chain):
	def __init__(self, in_ch, out_ch):
		layers = {}
		layers["c1"] = CBR(in_ch, 32, bn=True, sample='none-7')
		layers["c2"] = CBR(32, 64, bn=True, sample='down')
		layers["c3"] = CBR(64, 128, bn=True, sample='down')
		layers["c4"] = ResBlock(128, bn=True)
		layers["c5"] = ResBlock(128, bn=True)
		layers["c6"] = ResBlock(128, bn=True)
		layers["c7"] = ResBlock(128, bn=True)
		layers["c8"] = ResBlock(128, bn=True)
		layers["c9"] = ResBlock(128, bn=True)
		layers["c10"] = ResBlock(128, bn=True)
		layers["c11"] = ResBlock(128, bn=True)
		layers["c12"] = ResBlock(128, bn=True)
		layers["c13"] = CBR(128, 64, bn=True, sample='up')
		layers["c14"] = CBR(64, 32, bn=True, sample='up')
		layers["c15"] = CBR(32, out_ch, bn=True, sample='none-7', activation=F.tanh)
		super().__init__(**layers)

	def __call__(self, x):
		h = self.c1(x)
		h = self.c2(h)
		h = self.c3(h)
		h = self.c4(h)
		h = self.c5(h)
		h = self.c6(h)
		h = self.c7(h)
		h = self.c8(h)
		h = self.c9(h)
		h = self.c10(h)
		h = self.c11(h)
		h = self.c12(h)
		h = self.c13(h)
		h = self.c14(h)
		h = self.c15(h)
		return h

class Discriminator(chainer.Chain):
	def __init__(self, in_ch=3, n_down_layers=4):
		layers = {}
		self.n_down_layers = n_down_layers

		layers['c0'] = CBR(in_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
		base = 64

		for i in range(1, n_down_layers):
			layers['c' + str(i)] = CBR(base, base * 2, bn=True, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
			base *= 2

		layers['c' + str(n_down_layers)] = CBR(base, 1, bn=False, sample='none', activation=None, dropout=False, noise=True)

		super(Discriminator, self).__init__(**layers)

	def __call__(self, x_0):
		h = self.c0(x_0)
		d = self.__dict__
		for i in range(1, self.n_down_layers + 1):
			h = d['c' + str(i)](h)
		return h

class CycleGan():
	"""CycleGanのモデル.
	"""
	def __init__(self, name, in_ch, out_ch, batch_size, learning_rate_g=0.0002, learning_rate_d=0.0002, lambda1=10.0, lambda2=3.0, learning_rate_anneal=0.0, learning_rate_anneal_interval=1000):
		"""モデル初期化.
		# Args:
			name: モデル名、学習済みデータ保存時にファイル名に付与される.
			in_ch: 入力CH数.
			out_ch: 出力CH数.
			batch_size: バッチサイズ.
		"""
		assert in_ch == out_ch, "in_ch and out_ch must be same value."
		assert dsconf.MaxImageSize[0] == dsconf.MaxImageSize[1], "Image aspect ratio must be one."

		self.name = name

		# Set up a neural network to train
		self.gen_g = Generator_ResBlock_9(in_ch, out_ch)
		self.dis_x = Discriminator(out_ch)
		self.gen_f = Generator_ResBlock_9(in_ch, out_ch)
		self.dis_y = Discriminator(out_ch)

		# Setup an optimizer
		def make_optimizer(model, alpha=0.0002, beta1=0.5):
			optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
			optimizer.setup(model)
			return optimizer
		self.opt_g = make_optimizer(self.gen_g, alpha=learning_rate_g)
		self.opt_f = make_optimizer(self.gen_f, alpha=learning_rate_g)
		self.opt_x = make_optimizer(self.dis_x, alpha=learning_rate_d)
		self.opt_y = make_optimizer(self.dis_y, alpha=learning_rate_d)

		self.lambda1 = lambda1
		self.lambda2 = lambda2
		self.learning_rate_anneal = learning_rate_anneal
		self.learning_rate_anneal_interval = learning_rate_anneal_interval
		self.image_size = dsconf.MaxImageSize[0]
		self.iter = 0
		self.max_buffer_size = 50

		# gen_g の出力を蓄えておくリングバッファ
		self.ring_x = np.zeros((self.max_buffer_size, batch_size, in_ch, self.image_size, self.image_size), dtype=dnn.dtype)
		self.ring_x_len = 0 # バッファ内有効要素数
		self.ring_x_next = 0 # 次回バッファに書き込み時のインデックス
		# gen_f の出力を蓄えておくリングバッファ
		self.ring_y = np.zeros((self.max_buffer_size, batch_size, in_ch, self.image_size, self.image_size), dtype=dnn.dtype)
		self.ring_y_len = 0 # バッファ内有効要素数
		self.ring_y_next = 0 # 次回バッファに書き込み時のインデックス

	def save(self):
		"""ニューラルネットワークモデルの学習済みデータを保存する.
		"""
		dnn.SaveChains(
			self.name,
			{
				"gen_g": self.gen_g,
				"dis_x": self.dis_x,
				"gen_f": self.gen_f,
				"dis_y": self.dis_y
			})
		dnn.SaveOptimizers(
			self.name,
			{
				"opt_g": self.opt_g,
				"opt_f": self.opt_f,
				"opt_x": self.opt_x,
				"opt_y": self.opt_y
			})

	def load(self):
		"""ニューラルネットワークモデルの学習済みデータを読み込む.
		"""
		dnn.LoadChains(
			self.name,
			{
				"gen_g": self.gen_g,
				"dis_x": self.dis_x,
				"gen_f": self.gen_f,
				"dis_y": self.dis_y
			})
		dnn.LoadOptimizers(
			self.name,
			{
				"opt_g": self.opt_g,
				"opt_f": self.opt_f,
				"opt_x": self.opt_x,
				"opt_y": self.opt_y
			})

		# GPU用にモデルを変換する
		self.gen_g = dnn.ToGpu(self.gen_g)
		self.dis_x = dnn.ToGpu(self.dis_x)
		self.gen_f = dnn.ToGpu(self.gen_f)
		self.dis_y = dnn.ToGpu(self.dis_y)

	def train(self, x, t, saveEval, showEval):
		"""入力データと教師データを用いて学習を実行する.
		# Args:
			x: 入力データ. ※dnn.ToGpu() で変換済みでなければならない
			t: 教師データ. ※dnn.ToGpu() で変換済みでなければならない
			saveEval: 評価用画像を所定のディレクトリに保存するかどうか.
			showEval: 評価用画像を表示するかどうか.
		"""
		self.iter += 1

		w_in = self.image_size

		x = x if isinstance(x, Variable) else Variable(x)
		y = t if isinstance(t, Variable) else Variable(t)

		x_y = self.gen_g(x)
		x_y_copy = self.addXToRingAndGet(x_y.data)
		x_y_copy = Variable(x_y_copy)
		x_y_x = self.gen_f(x_y)

		y_x = self.gen_f(y)
		y_x_copy = self.addYToRingAndGet(y_x.data)
		y_x_copy = Variable(y_x_copy)
		y_x_y = self.gen_g(y_x)

		if self.learning_rate_anneal > 0 and self.iter % self.learning_rate_anneal_interval == 0:
			if self.opt_g.alpha > self.learning_rate_anneal:
				self.opt_g.alpha -= self.learning_rate_anneal
			if self.opt_f.alpha > self.learning_rate_anneal:
				self.opt_f.alpha -= self.learning_rate_anneal
			if self.opt_x.alpha > self.learning_rate_anneal:
				self.opt_x.alpha -= self.learning_rate_anneal
			if self.opt_y.alpha > self.learning_rate_anneal:
				self.opt_y.alpha -= self.learning_rate_anneal

		self.opt_g.zero_grads()
		self.opt_f.zero_grads()
		self.opt_x.zero_grads()
		self.opt_y.zero_grads()

		loss_dis_y_fake = loss_func_adv_dis_fake(self.dis_y(x_y_copy))
		loss_dis_y_real = loss_func_adv_dis_real(self.dis_y(y))
		loss_dis_y = loss_dis_y_fake + loss_dis_y_real
		# chainer.report({'loss': loss_dis_y}, self.dis_y)
		print('loss', loss_dis_y.data)

		loss_dis_x_fake = loss_func_adv_dis_fake(self.dis_x(y_x_copy))
		loss_dis_x_real = loss_func_adv_dis_real(self.dis_x(x))
		loss_dis_x = loss_dis_x_fake + loss_dis_x_real
		# chainer.report({'loss': loss_dis_x}, self.dis_x)
		print('loss', loss_dis_x.data)

		loss_dis_y.backward()
		loss_dis_x.backward()

		self.opt_y.update()
		self.opt_x.update()

		loss_gen_g_adv = loss_func_adv_gen(self.dis_y(x_y))
		loss_gen_f_adv = loss_func_adv_gen(self.dis_x(y_x))

		loss_cycle_x = self.lambda1 * loss_func_rec_l1(x_y_x, x)
		loss_cycle_y = self.lambda1 * loss_func_rec_l1(y_x_y, y)
		loss_gen = self.lambda2 * loss_gen_g_adv + self.lambda2 * loss_gen_f_adv + loss_cycle_x + loss_cycle_y
		loss_gen.backward()
		self.opt_f.update()
		self.opt_g.update()

		# chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
		# chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
		# chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
		# chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)
		print('loss_rec', loss_cycle_y.data)
		print('loss_rec', loss_cycle_x.data)
		print('loss_adv', loss_gen_g_adv.data)
		print('loss_adv', loss_gen_f_adv.data)

		if saveEval or showEval:
			img = np.zeros((x.data.shape[1], w_in * 2, w_in * 3), dtype=dnn.dtype)
			cells = [
				dnn.ToCpu(x.data[0]), dnn.ToCpu(x_y.data[0]), dnn.ToCpu(x_y_x.data[0]),
				dnn.ToCpu(y.data[0]), dnn.ToCpu(y_x.data[0]), dnn.ToCpu(y_x_y.data[0])
			]
			i = 0
			for r in range(2):
				rs = r * w_in
				for c in range(3):
					cs = c * w_in
					img[:, rs:rs + w_in, cs:cs + w_in] = cells[i]
					i += 1
			img = imgcnv.DnnToBgr(img)

			if saveEval:
				f = os.path.normpath(os.path.join(dnn.GetEvalDataDir(), "eval.png"))
				cv2.imwrite(f, img)
				uploader.Upload(f)
			if showEval:
				cv2.imshow("CycleGAN", img)

		return

	def test(self, x):
		"""学習済みのモデルを使用して変換を行う.
		# Args:
			x: 入力データ. ※dnn.ToGpu() で変換済みでなければならない
		# Returns:
			出力データchainer.Variable
		"""
		return None

	def addXToRingAndGet(self, data):
		"""指定値をリングバッファに追加し、リングバッファ内からランダムに選んだ値を返す.
		# Args:
			data: リングバッファに追加する値.
		# Returns:
			入力値またはリングバッファ内の値.
		"""
		# リングバッファキャパシティ
		n = len(self.ring_x)

		# とりあえず書き込む
		self.ring_x[self.ring_x_next, :] = dnn.ToCpu(data)
		self.ring_x_next = (self.ring_x_next + 1) % n
		if self.ring_x_len < n:
			self.ring_x_len += 1
			return data # バッファが埋まるまでは入力値をそのまま使用

		# 50%の確率で入力値をそのまま返す
		if np.random.rand() < 0.5:
			return data

		# リングバッファ内からランダムに選んで返す
		id = np.random.randint(0, self.max_buffer_size)
		return dnn.ToGpu(self.ring_x[id, :].reshape(data.shape[:2] + (self.image_size, self.image_size)))

	def addYToRingAndGet(self, data):
		"""指定値をリングバッファに追加し、リングバッファ内からランダムに選んだ値を返す.
		# Args:
			data: リングバッファに追加する値.
		# Returns:
			入力値またはリングバッファ内の値.
		"""
		# リングバッファキャパシティ
		n = len(self.ring_y)

		# とりあえず書き込む
		self.ring_y[self.ring_y_next, :] = dnn.ToCpu(data)
		self.ring_y_next = (self.ring_y_next + 1) % n
		if self.ring_y_len < n:
			self.ring_y_len += 1
			return data # バッファが埋まるまでは入力値をそのまま使用

		# 50%の確率で入力値をそのまま返す
		if np.random.rand() < 0.5:
			return data

		# リングバッファ内からランダムに選んで返す
		id = np.random.randint(0, self.max_buffer_size)
		return dnn.ToGpu(self.ring_y[id, :].reshape(data.shape[:2] + (self.image_size, self.image_size)))

	def addXToRingAndGet_ImShow(self, data):
		ret = self.addXToRingAndGet(data)
		img = np.zeros((data.shape[1], self.image_size, self.image_size * 2), dtype=dnn.dtype)
		img[:, :, :self.image_size] = dnn.ToCpu(data[0])
		img[:, :, self.image_size:] = dnn.ToCpu(ret[0])
		cv2.imshow("x", imgcnv.DnnToBgr(img))
		return ret

	def addYToRingAndGet_ImShow(self, data):
		ret = self.addYToRingAndGet(data)
		img = np.zeros((data.shape[1], self.image_size, self.image_size * 2), dtype=dnn.dtype)
		img[:, :, :self.image_size] = dnn.ToCpu(data[0])
		img[:, :, self.image_size:] = dnn.ToCpu(ret[0])
		cv2.imshow("y", imgcnv.DnnToBgr(img))
		return ret
