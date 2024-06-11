Ivan: x_redu [64,271,281] -> 
x_reduced = self.nonlin_in(tf.tensordot(x,self.W,axes=[[1],[0]], 
                                                            name='de-mix') + self.b_in)
x_redu [64,64,281] ->
x_reduced = tf.expand_dims(x_reduced, -2)

x_redu [64,64,1,281]
conv_ = self.nonlin_out(tf.nn.depthwise_conv2d(x_reduced, 
                                                self.filters, strides=[1,1,1,1],
                                                padding='SAME') + self.b)
        self.filters [filter_length,1,n_ls,1]
        Given a 4D input tensor ('NHWC' or 'NCHW' data formats) and a filter tensor of shape [filter_height, filter_width, in_channels, channel_multiplier] containing in_channels convolutional filters of depth 1, depthwise_conv2d applies a different filter to each input channel (expanding from 1 channel to channel_multiplier channels for each), then concatenates the results together. The output has in_channels * channel_multiplier channels.

conv_ [64,64,1,281]
pool [64,64,1,141]
return [64,64,141]

Ours:
self.conv = nn.Conv2d(n_ls, n_ls, kernel_size=(filter_length, 1), stride=(stride, 1),padding='same',groups=n_ls)
shape: [filter,1,n_ls,1]: [9,1,64,1]
    Depthwise convolution:
    When groups == in_channels and out_channels == K * in_channels
    In other words, for an input of size (N, C_{in}, L_{in})​,
    a depthwise convolution with a depthwise multiplier K can be performed with the arguments 
    $(C_{in}=C_{in}, C_{out}=C_{in} \times K, ..., \text{groups}=C_{in})$ 
__________________________________
x [64,271,281] ->
x_reduced = torch.einsum('bct,cs->bst', x, self.weights) 

x_redu [64,64,281] ->
x_reduced = x_reduced.unsqueeze(-2) #x_red.shap = batch, 64, 1, 281: batch, feature, groups,time

x_redu [64,64,1,281] -> 
conv_ = self.nonlin_out()(self.conv(x_reduced)).permute(0, 1, 3, 2)

conv_ [64,64,281,1]
pool [64,64,141,1]
return [64,64,141]