import chainer
import chainer.functions as F
import chainer.links as L
import dnn

# 参考元 https://github.com/pfnet-research/chainer-pix2pix

# U-net https://arxiv.org/pdf/1611.07004v1.pdf
# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
	def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
		self.bn = bn
		self.activation = activation
		self.dropout = dropout
		layers = {}
		w = chainer.initializers.Normal(0.02, dtype=dnn.dtype)
		# b = chainer.initializers.Constant(0, dtype=dnn.dtype)
		if sample == 'down':
			layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
		else:
			layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
		if bn:
			layers['batchnorm'] = L.BatchNormalization(ch1, dtype=dnn.dtype)
		super(CBR, self).__init__(**layers)

	def __call__(self, x):
		h = self.c(x)
		if self.bn:
			h = self.batchnorm(h, test=dnn.test)
		if self.dropout:
			h = F.dropout(h)
		if not (self.activation is None):
			h = self.activation(h)
		return h

class Encoder(chainer.Chain):
	def __init__(self, in_ch):
		layers = {}
		w = chainer.initializers.Normal(0.02, dtype=dnn.dtype)
		layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
		layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		super(Encoder, self).__init__(**layers)

	def __call__(self, x):
		hs = [F.leaky_relu(self.c0(x))]
		for i in range(1, 8):
			hs.append(self['c%d' % i](hs[i - 1]))
		return hs

class Decoder(chainer.Chain):
	def __init__(self, out_ch):
		layers = {}
		w = chainer.initializers.Normal(0.02, dtype=dnn.dtype)
		layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
		layers['c1'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
		layers['c2'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
		layers['c3'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=False)
		layers['c4'] = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
		layers['c5'] = CBR(512, 128, bn=True, sample='up', activation=F.relu, dropout=False)
		layers['c6'] = CBR(256, 64, bn=True, sample='up', activation=F.relu, dropout=False)
		layers['c7'] = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)
		super(Decoder, self).__init__(**layers)

	def __call__(self, hs):
		h = self.c0(hs[-1])
		for i in range(1, 8):
			h = F.concat([h, hs[-i - 1]])
			if i < 7:
				h = self['c%d' % i](h)
			else:
				h = self.c7(h)
		return h

class Discriminator(chainer.Chain):
	def __init__(self, in_ch, out_ch):
		layers = {}
		w = chainer.initializers.Normal(0.02, dtype=dnn.dtype)
		layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
		layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)
		super(Discriminator, self).__init__(**layers)

	def __call__(self, x_0, x_1):
		h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
		h = self.c1(h)
		h = self.c2(h)
		h = self.c3(h)
		h = self.c4(h)
		# h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
		return h

class Pix2Pix():
	"""Pix2Pixのモデル.
	"""
	def __init__(self, name, in_ch, out_ch):
		"""モデル初期化.
		# Args:
			name: モデル名、学習済みデータ保存時にファイル名に付与される.
			in_ch: 入力CH数.
			out_ch: 出力CH数.
		"""
		self.name = name

		# Set up a neural network to train
		self.enc = Encoder(in_ch=in_ch)
		self.dec = Decoder(out_ch=out_ch)
		self.dis = Discriminator(in_ch=in_ch, out_ch=out_ch)

		# Setup an optimizer
		def make_optimizer(model, alpha=0.0002, beta1=0.5):
			optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
			optimizer.setup(model)
			optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
			return optimizer
		self.opt_enc = make_optimizer(self.enc)
		self.opt_dec = make_optimizer(self.dec)
		self.opt_dis = make_optimizer(self.dis)

	def train(self, x, t):
		"""入力データと教師データを用いて学習を実行する.
		# Args:
			x: 入力データ. ※dnn.ToGpu() で変換済みでなければならない
			t: 教師データ. ※dnn.ToGpu() で変換済みでなければならない
		# Returns:
			出力データ chainer.Variable
		"""
		xVar = x if isinstance(x, chainer.Variable) else chainer.Variable(x)
		z = self.enc(xVar)
		x_out = self.dec(z)

		y_fake = self.dis(xVar, x_out)
		y_real = self.dis(xVar, t)

		self.opt_enc.update(self.loss_enc, self.enc, x_out, t, y_fake)
		for z_ in z:
			z_.unchain_backward()
		self.opt_dec.update(self.loss_dec, self.dec, x_out, t, y_fake)
		xVar.unchain_backward()
		x_out.unchain_backward()
		self.opt_dis.update(self.loss_dis, self.dis, y_real, y_fake)

		return x_out

	def test(self, x):
		"""学習済みのモデルを使用して変換を行う.
		# Args:
			x: 入力データ. ※dnn.ToGpu() で変換済みでなければならない
		# Returns:
			出力データchainer.Variable
		"""
		xVar = x if isinstance(x, chainer.Variable) else chainer.Variable(x)
		z = self.enc(xVar)
		x_out = self.dec(z)
		return x_out

	def save(self):
		"""ニューラルネットワークモデルの学習済みデータを保存する.
		"""
		dnn.SaveChains(self.name, {"enc": self.enc, "dec": self.dec, "dis": self.dis})
		dnn.SaveOptimizers(self.name, {"enc": self.opt_enc, "dec": self.opt_dec, "dis": self.opt_dis})

	def load(self):
		"""ニューラルネットワークモデルの学習済みデータを読み込む.
		"""
		dnn.LoadChains(self.name, {"enc": self.enc, "dec": self.dec, "dis": self.dis})
		dnn.LoadOptimizers(self.name, {"enc": self.opt_enc, "dec": self.opt_dec, "dis": self.opt_dis})

		# GPU用にモデルを変換する
		self.enc = dnn.ToGpu(self.enc)
		self.dec = dnn.ToGpu(self.dec)
		self.dis = dnn.ToGpu(self.dis)

	def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=1):
		batchsize, _, w, h = y_out.data.shape
		loss_rec = lam1 * (F.mean_absolute_error(x_out, t_out))
		loss_adv = lam2 * F.sum(F.softplus(-y_out)) / batchsize / w / h
		loss = loss_rec + loss_adv
		chainer.report({'loss': loss}, enc)
		print("loss enc=", loss.data)
		return loss

	def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=1):
		batchsize, _, w, h = y_out.data.shape
		loss_rec = lam1 * (F.mean_absolute_error(x_out, t_out))
		loss_adv = lam2 * F.sum(F.softplus(-y_out)) / batchsize / w / h
		loss = loss_rec + loss_adv
		chainer.report({'loss': loss}, dec)
		print("loss dec=", loss.data)
		return loss

	def loss_dis(self, dis, y_in, y_out):
		batchsize, _, w, h = y_in.data.shape
		L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
		L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
		loss = L1 + L2
		chainer.report({'loss': loss}, dis)
		print("loss dis=", loss.data)
		return loss
