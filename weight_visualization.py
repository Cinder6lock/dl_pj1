import os
import pickle
import torch
from pyvis.network import Network

saved_models_path = './saved_models'
output_dir = './interactive_graphs'
os.makedirs(output_dir, exist_ok=True)

def visualize_mlp_interactive(size_list, output_path):
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    
    # 添加节点
    for idx, size in enumerate(size_list):
        net.add_node(f"L{idx}", label=f"Layer {idx}\n({size} neurons)", shape='circle', size=20)

    # 添加边
    for idx in range(len(size_list) - 1):
        net.add_edge(f"L{idx}", f"L{idx+1}")

    net.barnes_hut()  # 自动调整布局
    net.show(output_path, notebook=False)  # 显示图形并保存为HTML文件
    print(f"Saved MLP interactive graph: {output_path}")

def visualize_cnn_interactive(state_dict, output_path):
    net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", notebook=False)

    conv_layers = []
    fc_layers = []
    for key in state_dict.keys():
        if 'conv' in key and 'weight' in key:
            conv_layers.append(key)
        if 'fc' in key and 'weight' in key:
            fc_layers.append(key)

    layers = []

    for idx, name in enumerate(conv_layers):
        W = state_dict[name]
        out_channels, in_channels, kH, kW = W.shape
        label = f"Conv2D\n{in_channels}→{out_channels}\n{kH}x{kW}"
        layers.append((f"C{idx}", label))

    for idx, name in enumerate(fc_layers):
        W = state_dict[name]
        out_features, in_features = W.shape
        label = f"Linear\n{in_features}→{out_features}"
        layers.append((f"F{idx}", label))

    # 添加节点
    for node_id, label in layers:
        net.add_node(node_id, label=label, shape='box', size=25)

    # 添加边
    for i in range(len(layers) - 1):
        net.add_edge(layers[i][0], layers[i+1][0])

    net.barnes_hut()
    net.show(output_path)
    print(f"Saved CNN interactive graph: {output_path}")

# 遍历saved_models文件夹
for filename in os.listdir(saved_models_path):
    filepath = os.path.join(saved_models_path, filename)
    model_name = filename.split('.')[0]
    output_path = os.path.join(output_dir, model_name + '.html')

    if filename.endswith('.pickle'):
        with open(filepath, 'rb') as f:
            param_list = pickle.load(f)
        size_list = param_list[0]  # 取size_list
        visualize_mlp_interactive(size_list, output_path)
    elif filename.endswith('.pth'):
        state_dict = torch.load(filepath, map_location='cpu')
        visualize_cnn_interactive(state_dict, output_path)