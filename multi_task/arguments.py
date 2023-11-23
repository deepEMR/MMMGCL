import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    # dataset 'eICU','mimiciii','mimiciii_CCU','mimiciii_NWARD','mimiciii_CSRU'
    parser.add_argument('--DS', dest='DS', default='mimiciii', help='Dataset')
    parser.add_argument('--gpu', type=str, default="4")
    parser.add_argument('--is_reinitialize_feature', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=64, help='')
    parser.add_argument('--num_experts', type=int, default=4)

    # NodeClassification Mortality LOS3 LOS7 Readmin
    parser.add_argument('--is_task_NodeClassification', type=int, default=0)
    parser.add_argument('--is_task_Mortality', type=int, default=1)
    parser.add_argument('--is_task_Readmin', type=int, default=1)
    parser.add_argument('--is_task_LOS3', type=int, default=1)
    parser.add_argument('--is_task_LOS7', type=int, default=1)

    # augmentation type 'dnodes','pedges','subgraph','mask_nodes',diff, 'substitute_nodes'
    # 'random2','random3','random4','None'
    parser.add_argument('--aug1', type=str, default='dnodes')
    parser.add_argument('--aug2', type=str, default='pedges')

    parser.add_argument('--is_g_vs_g', type=int, default=1)
    parser.add_argument('--is_g_vs_n', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    # evaluate_type:'svc','logistic','linearsvc', 'randomforest'
    parser.add_argument('--evaluate_type', type=str, default='logistic')
    # Adam,SGD
    parser.add_argument('--optimizer_type', type=str, default='Adam')
    # Encoder Model: GATConv, GCNConv, SAGEConv, MLP, TransformerConv
    parser.add_argument('--encoder_model', type=str, default='GATConv')
    # use_node_attr：where use attr trained by Glove
    # False True
    parser.add_argument('--use_node_attr', type=bool, default=True)
    # local_global_loss_ 中的参数： GAN  JSD X2 KL RKL DV H2 W1
    parser.add_argument('--measure', type=str, default='GAN')
    # focal nll
    parser.add_argument('--loss_type', type=str, default='nll')

    parser.add_argument('--exponent_main_g_1', type=float, default=3.0)
    parser.add_argument('--exponent_main_g_2', type=float, default=5.0)
    parser.add_argument('--exponent_main_g_3', type=float, default=6.0)
    parser.add_argument('--exponent_main_g_4', type=float, default=3.0)
    parser.add_argument('--exponent_main_n', type=float, default=3.0)
    parser.add_argument('--exponent_assist_gg', type=float, default=8.0)
    parser.add_argument('--exponent_assist_gn', type=float, default=9.0)

    return parser.parse_args()
