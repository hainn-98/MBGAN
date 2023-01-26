import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--top_k', default=10, type=int)
	parser.add_argument('--l2_reg', default=1e-2, type=float, help='l2 weight decay regularizer')
	parser.add_argument('--dimension', default=64, type=int, help='embedding size')
	parser.add_argument('--scale_list', default=2, type=int, help='scale size list')
	parser.add_argument('--head_attention_num', default=2, type=int, help='number of attention heads')

	parser.add_argument('--gen_lr', default=0.001, type=float)
	parser.add_argument('--dis_lr', default=0.001, type=float)
	parser.add_argument('--gen_batch_size', default=128, type=int)
	parser.add_argument('--dis_batch_size', default=64, type=int)
	parser.add_argument('--test_batch_size', default=64, type=int)
	parser.add_argument('--shoot', default=10, type=int)
	# parser.add_argument('--gen_num_blocks', default=2, type=int)
	# parser.add_argument('--dis_num_blocks', default=1, type=int)
	parser.add_argument('--gen_multiscale_layer', default=2, type=int)
	parser.add_argument('--dis_multiscale_layer', default=2, type=int)

	parser.add_argument('--gen_dropout_rate', default=0.2, type=float)
	parser.add_argument('--dis_dropout_rate', default=0.25, type=float)

	parser.add_argument('--generator_train_num', default=100, type=int)
	parser.add_argument('--discriminator_train_num', default=20, type=int)
	parser.add_argument('--gan_epoch_num', default=100, type=int)
	parser.add_argument('--num_pre_generator', default=400, type=int)
	parser.add_argument('--num_pre_discriminator', default=100, type=int)
	parser.add_argument('--gen_multiscale_head', default=8, type=int)
	parser.add_argument('--dis_multiscale_head', default=8, type=int)
	return parser.parse_args()
args = parse_args()
# jd2021
# args.user = 57721
# args.item = 4172
# tianchi
# args.user = 423423
# args.item = 874328
# tmall
# args.user = 805506#147894
# args.item = 584050#99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734

# swap user and item
# tem = args.user
# args.user = args.item
# args.item = tem

# args.decay_step = args.trn_num
# args.decay_step = args.item//args.batch
args.decay_step = args.trnNum//args.batch
