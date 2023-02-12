import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
from user_choice import user_interaction
from user_generation import generation
from sim_functions import get_issue_mapping, random_bootstrap
import math
import warnings
from scipy.special import rel_entr
import torch
import implicit
from scipy.sparse import coo_matrix
from MatrixFactorization import MatrixFactorization
warnings.filterwarnings("ignore")

# five different groups
prototypes = [
    "bystanders",
    "solid liberas",
    "oppty democrats",
    "market skeptic repub",
    "core conserv",
]

ptt2int = {
    "bystanders":0,
    "solid liberas":-2,
    "oppty democrats":-1,
    "market skeptic repub":1,
    "core conserv":2
}

# 14 different article topics
issues = ['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
          'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']

# whether clicking an article will change the user preference
has_influence = True

# whether the influence parameter is fixed value
fixed_influence = True

# influence parameter
topic_influenceG = 0.03

#whether implementing dynamic ranking calibration method
calibration = True
#whether implementing dynamic score calibration method
is_weight_change = True
# dynamic ranking calibration hyper parameter
linear_para = 2
# dynamic score calibration hyper parameter
weight_para = 8

def read_user_pkl(data):
    '''
    convert generated users to a dict object
    '''
    res = {}
    idx = 0

    for prototype, values in data.items():

        for matrix in values:

            res[idx] = {}
            res[idx]['ptt'] = prototype
            res[idx]['vec'] = matrix.flatten().tolist()
            idx += 1

    return res

def kl_divergence(p, q):
    '''
    calculate the kl divergence
    '''
    return np.sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))


def cal_history_dis(users_shown_history, shown_window):
    distribution = np.zeros(5)
    if len(users_shown_history)-shown_window < 0:
        start = 0
    else:
        start = len(users_shown_history)-shown_window

    for i in range(shown_window):
        distribution[users_shown_history[start+i]+2] += 1
    return distribution

def sigmoid(x):
    return 1 / (1 + math.exp(-5*x))

def linear(x):
    res = 0.5+linear_para*x
    if res>1:
        res = 1
    return res



# bootstrap, each user has 140 articals, 10 for each topic
def cf_algorithm(users, candidate_news, randomness, run_time, rank_folder, user_folder, bootstrap_folder, user_sequence_folder, hidden_dim=40, bootstrap_size=140, num_timestep=40000):
    '''
    collaborative-filtering algorithm
    '''
    users_clicked_pool = {}
    '''
    calibration list
    '''
    users_shown_history = {}
    original_users = users
    ori_users_distribution = {}
    users_exposure_history = {}
    users_clicked_history = {}
    users_clicked_time = {}

    issue_mapping = get_issue_mapping(candidate_news)

    # Create a dict from pk to column index
    # and create a dict from column index to pk
    pk2col = {}
    col2pk = {}

    for idx, pk in enumerate(list(candidate_news.keys())):
        pk2col[pk] = idx
        col2pk[idx] = pk

    history = {}
    for idx in range(len(users)):
        history[idx] = {}
        history[idx]['prototype'] = users[idx]['ptt']
        history[idx]['shown'] = []
        history[idx]['clicked'] = []

    #matrix is the user-article matrix
    matrix = np.zeros((len(users), len(candidate_news)))
    weight_matrix = np.ones((len(users), len(candidate_news)))
    group_weight_matrix = np.ones((5, 5))

    start = time.time()
    bootstrap_list = []

    # Initialize the bootstrap for each user
    for i in range(5):
        users_shown_history[i] = np.array([], dtype=int)
        ori_users_distribution[i] = np.zeros(5)
        users_clicked_time[i] = np.zeros(5)
        users_exposure_history[i] = np.array([], dtype=int)
        users_clicked_history[i] = np.array([], dtype=int)

    '''
    bootstrap
    '''
    for idx, user in sorted(users.items()):

        user_stance = user['ptt']
        user_stance_idx = ptt2int[user_stance]+2
        users_clicked_pool[idx] = set()

        pos_pks, neg_pks = random_bootstrap(
            user['vec'], issue_mapping, candidate_news, bootstrap_size)

        users_clicked_pool[idx].update(pos_pks)
        users_clicked_pool[idx].update(neg_pks)

        bootstrap_list.append(pos_pks)
        #update user-article matrix
        for pk in pos_pks:
            matrix[idx, pk2col[pk]] = 1

            top_news = candidate_news[pk]
            ori_users_distribution[user_stance_idx][int(top_news['source_partisan_score'])+2]+=1


    for i in range(5):
        ori_users_distribution[i] = ori_users_distribution[i]/np.sum(ori_users_distribution[i])

    print('ori_users_distribution--->\n', ori_users_distribution)
    end = time.time()

    bootstrap_file = bootstrap_folder + 'bootstrap_' + str(run_time) + '.pkl'
    bootstrap_data = pd.DataFrame(bootstrap_list)

    with open(bootstrap_file, "wb") as f:
        pickle.dump(bootstrap_data, f)

    print("user bootstrap spends %s" %str(end-start))

    # count is used to control the number of epoch
    count = 200
    users_len = len(users)
    new_matrix = matrix

    ave_rank = []
    user_time = -1
    user_sequence = []
    click_count = 0
    epoch = 140
    for _ in range(num_timestep):

        click_flag = False
        count += 1
        if _%800==0: epoch+=1
        if count >= 200:
            print('Last 200 users clicked ', click_count)
            click_count = 0
            user_time+= 1

            user_folder_inside = user_folder + '500users_' + str(run_time) +'epoch/'
            user_file_name = user_folder_inside + '500users_' + str(user_time) + ".pkl"
            with open(user_file_name, "wb") as f:
                pickle.dump(users, f)

            count = 0
            start = time.time()
            #MF
#
#             W, H, _ = non_negative_factorization(
#                 matrix, n_components=hidden_dim, init='random', random_state=random_seed, max_iter=250)
#             new_matrix = np.matmul(W, H)

#   torchnmf
#             if is_weight_change ==False:
#             matrix = torch.from_numpy(matrix)
#             matrix = matrix.cuda()
#             model = NMF(matrix.shape, rank=40)
#             model = model.cuda()
#
#             model.fit(matrix, max_iter=50, beta=2, l1_ratio=0.0, verbose=True)
#             new_matrix = model().cpu().detach().numpy()
#             matrix = matrix.cpu().detach().numpy()
#
#             end = time.time()
#             print("MF spends %s" %str(end-start))
#   implicit
# initialize a model
#             model = implicit.lmf.LogisticMatrixFactorization(factors=hidden_dim, use_gpu=True, random_state=random_seed)
#             model.fit(coo_matrix(matrix))
#             W = model.item_factors
#             H = model.user_factors
#
#             new_matrix = np.matmul(W, H.T)


            #matrix factorization parameter
            args = {
                'epoch':epoch,
                'display': 1,
                'learning_rate': 0.01,
                'reg': 0.0,
                'hidden': 40,
                'neg': 2,
                'bs': 256,
                'data': ''
            }
            args['num_user'] = matrix.shape[0]
            args['num_item'] = matrix.shape[1]
            start1 = time.time()
            r = np.where(matrix>0)
            matrix_list = []
            for i in range(len(r[0])):
                matrix_list.append([r[0][i],r[1][i]])

            print('r>0:  ', len(r[0]))
#             matrix_list
#             if is_weight_change:

            '''
            implementing dynamic score calibration
            '''
            if is_weight_change and _>=400:
                for i in range(5):
                    for j in range(5):
                        tmp_dis = users_clicked_time[i][j]/np.sum(users_clicked_time[i])
                        group_weight_matrix[i][j] = (ori_users_distribution[i][j]/tmp_dis)**weight_para
                        if group_weight_matrix[i][j]>12: group_weight_matrix[i][j]=12

            for i in range(500):
                for j in range(len(candidate_news)):
                    if(matrix[i][j]):
                        user_stance = users[i]['ptt']
                        user_stance_idx = ptt2int[user_stance]+2

                        candidate_pk=col2pk[j]
                        top_news = candidate_news[candidate_pk]
                        new_stance = int(top_news['source_partisan_score'])+2
                        weight_matrix[i][j] = group_weight_matrix[user_stance_idx][new_stance]


            matrix_df = pd.DataFrame(matrix_list, columns=['userId','itemId'])
            end1 = time.time()
            print("Inner MF spends %s" %str(end1-start1))

            model = MatrixFactorization(args, matrix_df)
            model.run(weight_matrix, is_weight_change)
            W, H = model.user_factors.weight, model.item_factors.weight
            W = W.detach().cpu().numpy()
            H = H.detach().cpu().numpy()
            new_matrix = np.matmul(W, H.T)


        #random choose a user
        rand_num = random.randint(0,users_len-1)
        rand_user = users[rand_num]
        rand_user_vec = rand_user['vec']

        user_sequence.append(rand_user['ptt'])

        row = new_matrix[rand_num][:]
        indices = np.argsort(row)[::-1]

        rec_count = 0

        candidate_pk_list = []
        article_num = 5
        candidate_num = 200
        start = time.time()

        user_stance = rand_user['ptt']

        if calibration==True:
#             start = time.time()
            '''
            calculate new ranking score after implementing dynamic ranking calibration
            '''
            user_stance_idx = ptt2int[user_stance]+2
            _range = np.max(row) - np.min(row)
            new_row =  (row - np.min(row)) / _range
            tmp_calibration_score = []
            index_count = 0


            tmp_users_exposure_distribution = np.ones(5)
            shown_window = 100
            end = shown_window+1


            # according to the clicked articles ditribution
            if len(users_clicked_history[user_stance_idx]) < shown_window:
                end = len(users_clicked_history[user_stance_idx])

            for j in range(1,end):
                tmp_users_exposure_distribution[users_clicked_history[user_stance_idx][-j]+2] +=1

            ckl_score = []
            erfa = 0.01
            pgu = ori_users_distribution[user_stance_idx]
            for i in range(5):
                inner_tmp = tmp_users_exposure_distribution
                inner_tmp[i]+=1
                inner_tmp = inner_tmp/np.sum(inner_tmp)

                qgu = (1-erfa)*inner_tmp + erfa*pgu
                ckl = kl_divergence(pgu, qgu)
                ckl_score.append(ckl)

            for col_idx in indices:
                acc_attribute = new_row[col_idx]
                news_stance = int(candidate_news[col2pk[col_idx]]['source_partisan_score'])
                if rec_count >= candidate_num:
                    break
                if col2pk[col_idx] not in users_clicked_pool[rand_num]:
                    rec_count += 1
                    lamda = linear(ckl_score[news_stance+2])
                    tmp_calibration_score.append((col_idx, (1-lamda)*acc_attribute-lamda*ckl_score[news_stance+2]))
#                     tmp_calibration_score.append((col_idx, cal_calibration(users_shown_history, ptt2int[user_stance]+2, acc_attribute, col2pk, col_idx, candidate_news, ori_users_distribution)))
                index_count+=1

            tmp_calibration_score.sort(key=lambda x:x[1], reverse=True)

            for i in range(article_num):
                candidate_pk_list.append(col2pk[tmp_calibration_score[i][0]])

#             endTime = time.time()
#             print("calibration spends %s" %str(endTime-start))

        else:
            for col_idx in indices:
                if rec_count >= article_num:
                    break
                if col2pk[col_idx] not in users_clicked_pool[rand_num]:
                    rec_count += 1
                    candidate_pk_list.append(col2pk[col_idx])

        end = time.time()

        rank = []
        start = time.time()


        for i in range(article_num):
            candidate_pk = candidate_pk_list[i]
            top_news = candidate_news[candidate_pk]
            users_clicked_pool[rand_num].add(candidate_pk)

            '''
            user interaction with the article
            flag equals true if the user click the recommended article
            '''
            flag = user_interaction(rand_user_vec, top_news, ranked=True)


            history[rand_num]['shown'].append([ _ ,  top_news])

            top_col_idx = pk2col[candidate_pk]

            user_stance_idx = ptt2int[user_stance]+2
            users_exposure_history[user_stance_idx] = np.append(users_exposure_history[user_stance_idx], int(top_news['source_partisan_score']))

            if flag:
                click_count+=1

                user_stance_idx = ptt2int[user_stance]+2
                users_clicked_history[user_stance_idx] = np.append(users_clicked_history[user_stance_idx], int(top_news['source_partisan_score']))
                users_clicked_time[user_stance_idx][int(top_news['source_partisan_score'])+2] += 1


                matrix[rand_num, top_col_idx] = 1
                history[rand_num]['clicked'].append([_,top_news])
#                 click_flag = True
                rank.append(i)
                if has_influence:
                    if fixed_influence:
                        users = fixed_update_user(users, rand_num, top_news)
#                     else:
#                         users = sn_random_update_user(users, users_change,rand_num, top_news)
#                 break

            else:
                if has_influence:
                    if fixed_influence:
                        users = fixed_update_user(users, rand_num, top_news, True)
                assert matrix[rand_num, top_col_idx] == 0
        end = time.time()


        ave_rank.append([rank, rand_user])


    file_name = rank_folder+"ave_rank_"+str(run_time)+".pkl"
    with open(file_name, "wb") as f:
        pickle.dump(np.asarray(ave_rank), f)

    file_name = user_sequence_folder+"user_sequence_"+str(run_time)+".pkl"
    with open(file_name, "wb") as f:
        pickle.dump(np.asarray(user_sequence), f)
    return history

def fixed_update_user(users, idx, top_news, negative=False):
    news_vec = top_news['topical_vector']
    topic_influence = topic_influenceG
    if negative:
        topic_influence = -topic_influence
    topics = np.count_nonzero(news_vec)
    fenmu = 29+topics*topic_influence
    for i in range(len(news_vec)):
        if(news_vec[i] == 1):
            users[idx]['vec'][i] = users[idx]['vec'][i] + topic_influence
        # normalization
        users[idx]['vec'][i] = 29/fenmu*users[idx]['vec'][i]
    return users

# def sn_random_update_user(users, users_change, index, top_news):
#     news_vec = top_news['topical_vector']
#     users[index]['vec'] = users[index]['vec'] + np.array(news_vec)*np.array(users_change[index]['vec'])
#     users[index]['vec'][users[index]['vec']<=0] = 1e-5
#     return users

def alg_cf(args, users, num_runs, candidate_news, output_file, run_time, rank_folder, user_folder, bootstrap_folder, user_sequence_folder):
    rst = {}

    start = time.time()
    history = cf_algorithm(users, candidate_news,
                           args.randomness, run_time, rank_folder, user_folder, bootstrap_folder, user_sequence_folder, num_timestep=args.num_recommends)
    history['users'] = users

    rst[0] = history
    end = time.time()


    with open(output_file, 'wb') as handle:
        pickle.dump(rst, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("The simulation of this run is complete!")

    return rst

def print_acc():
    prototypes = [
        "bystanders",
        "solid liberas",
        "oppty democrats",
        "market skeptic repub",
        "core conserv",
    ]
    acc_stance = {}
    num_people = 100
    run_time = 10
    for prototype in prototypes:
        all_sum_stance = 0
        for time in range(run_time):
            each_ite_list = []
            file_path = prototype+"/"+prototype+'_'+str(time)+'.pkl'

            with open(file_path,'rb') as f:
                data = pd.DataFrame(pickle.load(f))

            sum_clicked = 0
            sum_shown = 0
            for index, col in data.iteritems():
                sum_clicked += len(data[index]['clicked'])
                sum_shown += len(data[index]['shown'])

            acc = sum_clicked/sum_shown

            sum_stance = 0
            for index, col in data.iteritems():
                each_stance = 0
                for article in data[index]['clicked']:

                    each_stance += article[1]['source_partisan_score']
                if each_stance ==0: continue
                sum_stance += each_stance/len(data[index]['clicked'])
            all_sum_stance += sum_stance

        mean_stance = all_sum_stance/(num_people*run_time)
        acc_stance[prototype] = {'acc':acc, 'stance':mean_stance}

    print(acc_stance)

    df = pd.DataFrame(acc_stance)
    acc_array = df[0:1].to_numpy().reshape(5,)
    stance_array = df[1:2].to_numpy().reshape(5,)
    color = ['r','blue','green','yellow','orange']
    for i in range(len(acc_array)):
        plt.scatter(acc_array[i], stance_array[i],c=color[i], label=prototypes[i])
    plt.ylabel('Mean Political Stance')
    plt.xlabel('Mean Click Through Rate')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

if __name__ == "__main__":

#     mu = 0.03
#     sigma = 0.02

    parser = argparse.ArgumentParser(
        description="Parameters of simulation script")

    parser.add_argument(
        "--num_runs",
        default=1,
        help="the number of runs"
    )

    # control user number
    parser.add_argument(
        "--num_people",
        default=500,
        help="the number of people per run"
    )

    parser.add_argument(
        "--randomness",
        default=0.0,
        help="the randomness parameter coupled with CF algorithm"
    )

    parser.add_argument(
        "--dataset_type",
        default="balance",
        help="the type of dataset")

    #article data
    parser.add_argument(
        "--article_file",
        default="../data/balanced_political_news_40k_v2.pkl",
        help="the dataset of political articles",
    )

    parser.add_argument(
        "--pew_study",
        default="../data/pew0423.csv",
        help="the file of pew study to generate prototypes",
    )

    # control total recommendations number
    parser.add_argument(
        "--num_recommends",
        default=40000,
        help="the number of recommendations for each user",
    )

    # user data
    parser.add_argument(
        "--users",
        default="../data/500_rescale.pkl",
        help="The path of users file",
    )

    parser.add_argument(
        "--output_folder",
        default="../",
        help="the output folder of this experiment",
    )

    parser.add_argument(
        "--output_key",
        default="shown",
        help="the keyword to select the output to analyze. Could be clicked or shown",
    )

    args = parser.parse_args()

    with open(args.article_file, "rb") as handle:
        candidate_news = pickle.load(handle)


    now = datetime.now()
    print("now =", now)

    output_folder = (
        args.output_folder
        + "CF_"
        + time.strftime("%Y%m%d-%H%M%S")
        + "_R="
        + str(args.randomness)
    )



    if has_influence:
        if fixed_influence:
            output_folder += "_fixed"
            if topic_influenceG != 0:
                output_folder += "_influence=" + str(topic_influenceG)
#         else:
#             output_folder += '_mu:' + str(mu) + '_sigma:' + str(sigma)

    if calibration:
        output_folder += '_calibration=' + str(linear_para)

    if is_weight_change:
        output_folder += '_change_weight=' + str(weight_para)
    output_folder += "/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cfmf_folder = output_folder + "cfmf/"
    if not os.path.exists(cfmf_folder):
        os.makedirs(cfmf_folder)

    rank_folder = output_folder + "ave_rank/"
    if not os.path.exists(rank_folder):
        os.makedirs(rank_folder)

    user_folder = output_folder + "users/"
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    user_sequence_folder = output_folder + "user_sequence/"
    if not os.path.exists(user_sequence_folder):
        os.makedirs(user_sequence_folder)



    bootstrap_folder = output_folder + "bootstrap/"
    if not os.path.exists(bootstrap_folder):
        os.makedirs(bootstrap_folder)

    # control repeated run time
    run_time = 5

    for i in range(run_time):
        user_folder_inside = user_folder + '500users_' + str(i) +'epoch/'
        if not os.path.exists(user_folder_inside):
            os.makedirs(user_folder_inside)

    for i in range(run_time):

        if args.users:
            print("Loading the users' profile")
            with open(args.users, "rb") as f:
                users = pickle.load(f)

        else:
            record = generation(args.num_people, args.pew_study, equal=True)
            users = read_user_pkl(record)

            file_name = "./data/" + str(args.num_people) + ".pkl"
            with open(file_name, "wb") as f:
                pickle.dump(users, f)

        start = time.time()

        output_file = cfmf_folder + "cfmf"+"_"+ str(i)+".pkl"

        record = alg_cf(args, users, args.num_runs, candidate_news, output_file, i, rank_folder, user_folder, bootstrap_folder, user_sequence_folder)


        user_file_name = user_folder + '500users_' + str(i) + ".pkl"
        with open(user_file_name, "wb") as f:
            pickle.dump(users, f)

        prototype_record = {}

        for synth in prototypes:
            prototype_record[synth] = {}

        for key, value in record.items():
            for idx in range(args.num_people):
                prototype = value[idx]['prototype']
                length = len(prototype_record[prototype])
                prototype_record[prototype][length] = value[idx]

        for key, value in prototype_record.items():
            inside_output_folder = output_folder + key + "/"
            if not os.path.exists(inside_output_folder):
                os.makedirs(inside_output_folder)

            file_name = inside_output_folder + key + "_" + str(i) + ".pkl"
            with open(file_name, "wb") as file_handle:
                pickle.dump(value, file_handle)

        end = time.time()
        print("This is %s-th run, spending %s seconds" %
              (str(i), str(end-start)))
