import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import *
from tqdm import tqdm


import seaborn as sns

#mpl.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--nS", default=5, type=int)
parser.add_argument("--nA", default=5, type=int)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--lr_decay", default=0.95, type=float)
parser.add_argument("--n", default=5, type=int)
parser.add_argument("--iter", default=128)
args = parser.parse_args()

nS, nA, gamma = args.nS, args.nA, args.gamma
ntrain = args.n
ntest = 4 * args.n

d_0 = np.ones(nS) / nS

save_dir = './heter-plot-QAvg-random_MDP'

outer_iter_num = 16000

E_lst = (1, 2, 4, 8, 16, 32)
kappa_lst = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

E_size = len(E_lst) + 1
kappa_lst_len = len(kappa_lst)

# The length of the separator symbol
separation_len = 50

Seed = 101

data = [[] for i in range(E_size)]
iter_list = [3, 7, 15, 31, 63, 127]


def experiment():
    # Store the trained model parameters
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.random.seed(Seed)
    seeds = np.random.randint(low=0, high=1000000, size=outer_iter_num)

    total_MEObj_lst = np.zeros((outer_iter_num, kappa_lst_len, E_size, args.iter))
    total_test_MEObj_lst = np.zeros((outer_iter_num, kappa_lst_len, E_size, args.iter))

    total_test_center_MEObj_lst = np.zeros((outer_iter_num, kappa_lst_len, E_size, args.iter))

    total_MEObj_base_lst = np.zeros((outer_iter_num, kappa_lst_len, args.iter))  # Baseline
    total_MEObj_base_avg_lst = np.zeros((outer_iter_num, kappa_lst_len, args.iter))  # Baseline2
    total_test_MEObj_base_lst = np.zeros((outer_iter_num, kappa_lst_len, args.iter))  # Baseline
    total_test_MEObj_base_avg_lst = np.zeros((outer_iter_num, kappa_lst_len, args.iter))  # Baseline2

    for count in tqdm(range(outer_iter_num)):
        seed = seeds[count]
        R = np.random.uniform(low=0, high=1, size=(nS, nA))
        Q_init = np.random.uniform(size=(nS, nA))
        mix_Ps = generate_mix_Ps_Orth(n=ntrain + ntest, nS=nS, nA=nA, eps_lst=kappa_lst, seed=seed)

        # Training was performed according to different environmental heterogeneity
        for i in range(kappa_lst_len):
            Ps = mix_Ps[i]

            center_P = [Ps[0]]

            train_P = Ps[:ntrain]
            test_P = Ps[ntrain:]

            MEObj_lst = np.zeros((E_size, args.iter))
            test_MEObj_lst = np.zeros((E_size, args.iter))

            test_center_MEObj_lst = np.zeros((E_size, args.iter))

            MEObj_base_lst = np.zeros((args.iter))  # Baseline
            MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2
            test_MEObj_base_lst = np.zeros((args.iter))  # Baseline
            test_MEObj_base_avg_lst = np.zeros((args.iter))  # Baseline2

            # QAvg with different Es
            for E_num in range(E_size-1):
                E = E_lst[E_num]
                Q = Q_init.copy()
                lr = args.lr
                for e in range(args.iter // E):
                    # If the decrease is too small or negative, reduce the lr ? 变化太小不是应该加大学习率吗 ？此处尝试更改
                    if e > 2 and (MEObj_lst[E_num][(e - 1)*E] - MEObj_lst[E_num][(e - 2)*E]) < MEObj_lst[E_num][(e - 2)*E] * 1e-3:
                        lr = lr * args.lr_decay
                        # lr = lr * (2 - args.lr_decay)

                    V = QtoV(Q)
                    pi = QtoPolicy(Q)

                    center_ME_Obj = evaluate_MEObj_from_policy(pi, R, center_P, d_0, gamma)

                    ME_Obj = evaluate_MEObj_from_policy(pi, R, train_P, d_0, gamma)
                    ME_Obj_test = evaluate_MEObj_from_policy(pi, R, test_P, d_0, gamma)
                    for t in range(E):
                        MEObj_lst[E_num][e * E + t] = ME_Obj
                        test_MEObj_lst[E_num][e * E + t] = ME_Obj_test

                        test_center_MEObj_lst[E_num][e * E + t] = center_ME_Obj

                    Qs = []
                    for t in range(ntrain):
                        Vi = V.copy()
                        for _ in range(E):
                            delta_Vi, _ = value_iteration(Vi, train_P[t], R, gamma)
                            Vi = (1 - lr) * Vi + lr * delta_Vi
                        Qi = VtoQ(Vi, train_P[t], R, gamma)
                        Qs.append(Qi)
                    Q = sum(Qs) / ntrain

            # Baseline: Train every agent separately, i.e. do not merge
            lr = args.lr
            Q = Q_init.copy()
            Qs = [Q.copy() for _ in range(ntrain)]
            Q_avg = Q
            for e in range(args.iter):
                if e > 2 and (MEObj_lst[-1][e - 1] - MEObj_lst[-1][e - 2]) < MEObj_lst[-1][e - 2] * 1e-3:
                    lr = lr * args.lr_decay

                pi_avg = QtoPolicy(Q_avg)

                ME_Obj = evaluate_MEObj_from_policy(pi_avg, R, train_P, d_0, gamma)
                ME_Obj_test = evaluate_MEObj_from_policy(pi_avg, R, test_P, d_0, gamma)

                center_ME_Obj = evaluate_MEObj_from_policy(pi_avg, R, center_P, d_0, gamma)

                MEObj_lst[-1][e] = ME_Obj
                test_MEObj_lst[-1][e] = ME_Obj_test

                test_center_MEObj_lst[-1][e] = center_ME_Obj

                for i in range(ntrain):
                    Qi = Qs[i]
                    Vi = QtoV(Qi)
                    delta_Vi, _ = value_iteration(Vi, train_P[i], R, gamma)
                    Vi = (1 - lr) * Vi + lr * delta_Vi
                    # Vi = Vi - lr * delta_Vi
                    Qi = VtoQ(Vi, train_P[i], R, gamma)
                    Qs[i] = Qi
                Q_avg = sum(Qs) / ntrain

                best_ME_Obj = -inf
                best_label = -1
                avg_ME_Obj = 0
                avg_ME_Obj_test = 0
                ME_Obj_tests = []
                for i in range(ntrain):
                    Qi = Qs[i]
                    pi_i = QtoPolicy(Qi)
                    ME_Obj_i = evaluate_MEObj_from_policy(pi_i, R, train_P, d_0, gamma)
                    avg_ME_Obj += ME_Obj_i
                    if ME_Obj_i > best_ME_Obj:
                        best_ME_Obj = ME_Obj_i
                        best_label = i
                    # Test
                    ME_Obj_i_test = evaluate_MEObj_from_policy(pi_i, R, test_P, d_0, gamma)
                    avg_ME_Obj_test += ME_Obj_i_test
                    ME_Obj_tests.append(ME_Obj_i_test)
                MEObj_base_lst[e] = best_ME_Obj
                MEObj_base_avg_lst[e] = avg_ME_Obj / ntrain
                test_MEObj_base_avg_lst[e] = avg_ME_Obj_test / ntrain
                test_MEObj_base_lst[e] = ME_Obj_tests[best_label]
            MEObj_lst = MEObj_lst - 0.5 / (1 - gamma)
            test_MEObj_lst = test_MEObj_lst - 0.5 / (1 - gamma)
            MEObj_base_avg_lst = MEObj_base_avg_lst - 0.5 / (1 - gamma)
            test_MEObj_base_avg_lst = test_MEObj_base_avg_lst - 0.5 / (1 - gamma)
            MEObj_base_lst = MEObj_base_lst - 0.5 / (1 - gamma)
            test_MEObj_base_lst = test_MEObj_base_lst - 0.5 / (1 - gamma)

            test_center_MEObj_lst = test_center_MEObj_lst - 0.5 / (1 - gamma)

            total_MEObj_lst[count][i] = MEObj_lst
            total_MEObj_base_lst[count][i] = MEObj_base_lst
            total_MEObj_base_avg_lst[count][i] = MEObj_base_avg_lst
            total_test_MEObj_lst[count][i] = test_MEObj_lst
            total_test_MEObj_base_lst[count][i] = test_MEObj_base_lst
            total_test_MEObj_base_avg_lst[count][i] = test_MEObj_base_avg_lst

            total_test_center_MEObj_lst[count][i] = test_center_MEObj_lst

    np.save(save_dir + '/MEOBj_lst.npy', total_MEObj_lst)
    np.save(save_dir + '/MEOBj_base_lst.npy', total_MEObj_base_lst)
    np.save(save_dir + '/MEOBj_base_avg_lst.npy', total_MEObj_base_avg_lst)

    np.save(save_dir + '/test_MEOBj_lst.npy', total_test_MEObj_lst)
    np.save(save_dir + '/test_MEOBj_base_lst.npy', total_test_MEObj_base_lst)
    np.save(save_dir + '/test_MEOBj_base_avg_lst.npy', total_test_MEObj_base_avg_lst)

    np.save(save_dir + '/test_center_MEOBj_lst.npy', total_test_center_MEObj_lst)
    return


def report():
    print('Save dir=' + save_dir)
    coff = np.sqrt(outer_iter_num)
    total_center_MEOBj_lst = np.load(save_dir + '/test_center_MEOBj_lst.npy')
    for E_num in range(E_size):
        print('-' * separation_len)
        if E_num == E_size - 1:
            print('E=Inf:')
        else:
            print(f'E={E_lst[E_num]}:')
        center_MEOBj_lst = total_center_MEOBj_lst[:, :, E_num, :]
        center_MEObj_mean_lst = []
        center_MEObj_std_lst = []
        for i in range(kappa_lst_len):
            center_MEObj_mean_lst.append(np.average(center_MEOBj_lst[:, i, :], axis=0))
            center_MEObj_std_lst.append(np.std(center_MEOBj_lst[:, i, :], axis=0) / coff)
            for j in range(len(center_MEObj_mean_lst[i])):
                # 研究 k = 0.4 时 E 对收敛速率的影响
                if i == 2 and j in iter_list:
                    data[E_num].append(center_MEObj_mean_lst[i][j])
        # Here, kappailon is kappa in the paper
        for i in range(kappa_lst_len):
            print(f'Center kappa={kappa_lst[i]}: Mean={center_MEObj_mean_lst[i][-1]}, '
                  f'Std={center_MEObj_std_lst[i][-1]}')
    return

def draw(data):
    sns.set()
    np.random.seed(0)
    f, ax = plt.subplots(figsize=(9, 6))
    
    #heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
    #参数annot=True表示在对应模块中注释值
    # 参数linewidths是控制网格间间隔
    #参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
    #参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
    sns.heatmap(data, ax=ax,vmin=0,vmax=1,cmap='YlOrRd',annot=True,linewidths=2,cbar=None)
    
    ax.set_ylabel('E')  #设置纵轴标签
    ax.set_xlabel('Iteration')  #设置横轴标签
    
    #设置坐标字体方向，通过rotation参数可以调节旋转角度
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    
    plt.show()
    
    return


if __name__ == '__main__':
    experiment()
    report()
    draw(data)
