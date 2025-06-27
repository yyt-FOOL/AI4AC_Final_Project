import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import pickle
import os
import glob
import seaborn as sns
from unicore.data import Dictionary

def pad_attention_list(attention_list, pad_value=0):
    max_neighbors = max(att.shape[1] for att in attention_list)
    padded = []
    for att in attention_list:
        pad_width = max_neighbors - att.shape[1]
        if pad_width > 0:
            att_padded = np.pad(att, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_value)
        else:
            att_padded = att
        padded.append(att_padded)
    return np.concatenate(padded, axis=0)

def get_result(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    predict = []
    target = []
    src_tokens = []
    # node_attentions=[]
    for item in data:
        predict.extend(item['predict'].reshape(-1).tolist())
        target.extend(item['target'].reshape(-1).tolist())
        src_token = item["src_token"][item["select_atom"]==1]
        src_token = src_token.detach().cpu().numpy().tolist()
        src_tokens.extend(src_token)
        # node_attention = item['node_attention'].cpu().numpy()
    # node_attentions = pad_attention_list(node_attentions)
    return target, predict, src_tokens

def reg_metrics(target, predict):
    r2 = r2_score(target, predict)
    mae = mean_absolute_error(target, predict)
    mse = mean_squared_error(target, predict)
    rmse = math.sqrt(mse)
    return r2, mae, mse, rmse

def plot_metrics(target, predict, save_path=None, element="All"):
    r2, mae, mse, rmse = reg_metrics(target, predict)
    plt.figure(figsize=(8, 8))
    plt.scatter(target, predict, color='blue', alpha=0.5)
    xy_max = max(max(target), max(predict))
    xy_min = min(min(target), min(predict))
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color='black', linestyle='--')
    plt.xlim(xy_min, xy_max)  
    plt.ylim(xy_min, xy_max)  
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted\nMAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}'.format(mae, rmse, r2))
    # plt.legend()
    if os.path.exists(save_path):
        fig_path = os.path.join(save_path, f'result_{element}.png')
        plt.savefig(fig_path)
    plt.show()


# def plot_attention_distribution(attention, tokens, dictionary, save_path, element="All"):
#     """
#     Plots heatmaps of attention (or entropy) values for selected atoms, optionally grouped by element type.

#     Args:
#         attention (np.ndarray or torch.Tensor): shape [M, N] -- M selected nodes × N neighbors
#         tokens (List[int]): token types for the selected M nodes (e.g., atom indices)
#         dictionary (dict): id-to-element mapping, e.g., {0: 'H', 1: 'C', 2: 'O', ...}
#         save_path (str): folder to save the figure
#         element (str or List[str]): if 'All', plots for all element types found in tokens; otherwise only specified element(s)
#     """
#     if isinstance(attention, torch.Tensor):
#         attention = attention.detach().cpu().numpy()

#     tokens = np.array(tokens)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     # 构建 token → element 名称的映射数组
#     element_names = np.array([dictionary.get(t, f"UNK({t})") for t in tokens])

#     if element == "All":
#         target_elements = sorted(set(element_names))
#     elif isinstance(element, str):
#         target_elements = [element]
#     elif isinstance(element, (list, tuple)):
#         target_elements = element
#     else:
#         raise ValueError("element should be 'All', a string, or a list of strings")

#     for elem in target_elements:
#         selected_mask = element_names == elem
#         attn_elem = attention[selected_mask]
#         if attn_elem.shape[0] == 0:
#             print(f"[Warning] No attention entries for element: {elem}")
#             continue

#         plt.figure(figsize=(10, 6))
#         sns.heatmap(attn_elem, cmap="Purples", cbar=True, xticklabels=False, yticklabels=False)
#         plt.title(f"Attention Heatmap ({elem})")
#         plt.xlabel("Neighbor index")
#         plt.ylabel("Selected node index")
#         plt.tight_layout()

#         filename = f"attention_heatmap_{elem}.png"
#         plt.savefig(os.path.join(save_path, filename))
#         plt.close()

#         print(f"[{elem}] Attention Heatmap saved to {save_path}")
#         print(f"  Attention stats: mean={np.mean(attn_elem):.4f}, std={np.std(attn_elem):.4f}, max={np.max(attn_elem):.4f}")


def main(args):
    dictionary = Dictionary.load(args.dict)
    if args.mode == 'cv':
        target = 0
        src_tokens = 0
        all_predict = []
        # all_node_attention=[]
        for folder in os.listdir(args.path):
            folder_path = os.path.join(args.path, folder)
            if os.path.isdir(folder_path):
                pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
                filename = pkl_files[0]
                target, predict, src_tokens= get_result(filename)
                all_predict.append(predict)
                # all_node_attention.append(node_attention)
                plot_metrics(target, predict, folder_path)
                # plot_attention_distribution(node_attention, src_tokens, dictionary, folder_path)
                r2, mae, mse, rmse = reg_metrics(target, predict)
                print(f'metric of {filename}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

                elemenets = set(src_tokens)
                if len(elemenets) > 1:
                    for element in elemenets:
                        element_targets = np.array(target)[np.array(src_tokens)==element]
                        element_predicts = np.array(predict)[np.array(src_tokens)==element]
                        r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                        plot_metrics(target, predict, folder_path, dictionary[element])
                        # plot_attention_distribution(node_attention, src_tokens, dictionary, folder_path,dictionary[element])
                        print(f'metric of {dictionary[element]}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

        if all_predict:
            mean_predict = np.mean(np.vstack(all_predict), axis=0)
            plot_metrics(target, mean_predict, args.path)
            # plot_attention_distribution(node_attention, src_tokens, dictionary, args.path)
            r2, mae, mse, rmse = reg_metrics(target, mean_predict)
            print(f'metric of mean\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
            elemenets = set(src_tokens)
            if len(elemenets) > 1:
                for element in elemenets:
                    element_targets = np.array(target)[np.array(src_tokens)==element]
                    element_predicts = np.array(mean_predict)[np.array(src_tokens)==element]
                    r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                    plot_metrics(element_targets, element_predicts, args.path, dictionary[element])
                    # plot_attention_distribution(node_attention, src_tokens, dictionary, args.path,dictionary[element])
                    print(f'metric of {dictionary[element]}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')

            mean_path = os.path.join(args.path, 'mean.pkl')
            mean_data = {'target': target, 'predict': mean_predict}
            with open(mean_path, 'wb') as file:
                pickle.dump(mean_data, file)

    else :
        pkl_files = glob.glob(os.path.join(args.path, "*.pkl"))
        filename = pkl_files[0]
        target, predict, src_tokens= get_result(filename)
        r2, mae, mse, rmse = reg_metrics(target, predict)
        print(f'metric of {filename}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        plot_metrics(target, predict, args.path)
        # plot_attention_distribution(node_attention, src_tokens, dictionary, args.path)
        elemenets = set(src_tokens)
        if len(elemenets) > 1:
            for element in elemenets:
                element_targets = np.array(target)[np.array(src_tokens)==element]
                element_predicts = np.array(predict)[np.array(src_tokens)==element]
                r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                plot_metrics(target, predict, args.path, dictionary[element])
                # plot_attention_distribution(node_attention, src_tokens, dictionary, args.path, dictionary[element])
                print(f'metric of {dictionary[element]}\n: R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--path', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--dict', type=str)
    args = parser.parse_args()
    main(args)