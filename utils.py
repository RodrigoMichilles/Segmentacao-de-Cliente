import sys
from datetime import datetime
# Warnings.
from warnings import filterwarnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

filterwarnings('ignore')

def describe_categorical(df, feature):
    print(f"{feature}\n{'-'*40}\nUnique values: {df[feature].nunique()}")
    print("Value counts:")
    for value, count in df[feature].value_counts(normalize=True).items():
        print(f"- {value}: {count:.2%}")
    print()

class CustomException(Exception):
    def __init__(self, message, sys):
        super().__init__(f"An error occurred: {message}\n{sys.exc_info()}")

def sns_plots(data, features, plot_type='hist', y_feature=None, x_feature=None,
              hue=None, palette=None, style='whitegrid', show_outliers=False, kde=False, outliers=False):
    
   
    try:
        sns.set_theme(style=style)
        num_features = len(features)
        num_rows = (num_features + 2) // 3  # Ceiling division for correct number of rows

         # Define um palette padrão, se não for especificado
        if palette is None:
            palette = sns.color_palette("Set2")

        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5 * num_rows))
        axes = axes.flatten() 

        for i, feature in enumerate(features):
            ax = axes[i]

            if plot_type == 'count':
                sns.countplot(data=data, x=feature, hue=hue, ax=ax, palette=palette)
                for container in ax.containers:
                    ax.bar_label(container)
            elif plot_type == 'bar':
                sns.barplot(data=data, x=x_feature, y=feature if y_feature is None else y_feature, hue=hue, ax=ax, ci=None, palette=palette)
                for container in ax.containers:
                    ax.bar_label(container)
            elif plot_type == 'boxplot':
                sns.boxplot(data=data, x=x_feature, y=feature, showfliers=show_outliers, ax=ax, palette=palette)
            
            elif plot_type == 'barplot':
                # Plotting barplot and adding the averages at the top of each bar.
                ax = sns.barplot(data=data, x=x_feature, y=feature, hue=hue, ax=ax, ci=None, palette=palette)
                for container in ax.containers:
                    ax.bar_label(container)

            elif plot_type == 'outliers':
                # Plotting univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, palette=palette)

            elif plot_type == 'hist':
                sns.histplot(data=data, x=feature, hue=hue, kde=kde, ax=ax, palette=palette)
            else:
                raise ValueError("Invalid plot_type. Choose from 'hist', 'count', 'bar', or 'boxplot'.")

            ax.set_title(feature)
            ax.set_xlabel('')

        # Remove unused subplots
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show() # explicitly show the plot.
    except Exception as e:
        raise CustomException(e, sys)
    



def analyze_outliers(df, features, verbose=True):
    
    # Lista para armazenar resultados
    results = []
    # Dicionários para armazenar contagem e índices dos outliers
    outlier_indices = {}
    outlier_counts = {}
    total_outliers = 0

    # Verifica se todas as colunas especificadas existem no DataFrame
    for column in features:
        if column not in df.columns:
            raise ValueError(f"Coluna '{column}' não encontrada no DataFrame")

    for column in features:
        # Calcula Q1, Q3 e IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define limites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identifica outliers
        outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_indices[column] = df[outliers_mask].index.tolist()
        outliers = df[outliers_mask][column]
        n_outliers = len(outliers)
        perc_outliers = (n_outliers / len(df)) * 100
        outlier_counts[column] = n_outliers
        total_outliers += n_outliers

        # Adiciona resultados à lista
        results.append({
            'feature': column,
            'quantidade_outliers': n_outliers,
            'porcentagem_outliers': round(perc_outliers, 2),
            'limite_inferior': round(lower_bound, 2),
            'limite_superior': round(upper_bound, 2),
            'Q1': round(Q1, 2),
            'Q3': round(Q3, 2),
            'IQR': round(IQR, 2)
        })

    # Cria DataFrame com os resultados
    results_df = pd.DataFrame(results)

    # Reordena as colunas para melhor visualização
    results_df = results_df[[
        'feature', 'quantidade_outliers', 'porcentagem_outliers',
        'limite_inferior', 'limite_superior',
        'Q1', 'Q3', 'IQR'
    ]]

    if verbose:
        print(f'Total de outliers no dataset: {total_outliers}')
        print()
        print('Número (porcentagem) de outliers por feature:')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(df)*100, 2)}%)')

    return outlier_indices, outlier_counts, total_outliers



    

def feature_engineering(data):

    try:
        # cria uma copia do df
        feat_eng_cust = data.copy()

        # renomeia algumas features
        feat_eng_cust=feat_eng_cust.rename(columns={"MntWines": "Wines",
                                 "MntFruits":"Fruits",
                                 "MntMeatProducts":"Meat",
                                 "MntFishProducts":"Fish",
                                 "MntSweetProducts":"Sweets",
                                 "MntGoldProds":"Gold"})
        
        # transforma os nomes das colunas em minusculo
        feat_eng_cust.columns = [x.lower() for x in  feat_eng_cust.columns]

        # Converte dt_customer em datatime
        feat_eng_cust['dt_customer'] = pd.to_datetime(feat_eng_cust['dt_customer'], format='%d-%m-%Y')

        # Dropa outliers e algumas informações inconsistentes
        numerical_features = feat_eng_cust.select_dtypes('number').columns.to_list()
        outlier_indexes, _, _ = analyze_outliers(df=feat_eng_cust, features=numerical_features, verbose=False)
        to_drop_indexes = outlier_indexes['income'] + outlier_indexes['year_birth']
        feat_eng_cust.drop(to_drop_indexes, inplace=True)

        # Feature engineering.
        
        # Mesclando categorias de educação e estado civil
        feat_eng_cust['education'] = feat_eng_cust['education'].map({'Graduation': 'Graduate', 'PhD': 'Postgraduate', 'Master': 'Postgraduate', '2n Cycle': 'Undergraduate', 'Basic': 'Undergraduate'})
        feat_eng_cust['marital_status'] = feat_eng_cust['marital_status'].map({'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single', 'Married': 'Partner', 'Together': 'Partner'})


        # criando a feature dependents.
        feat_eng_cust['dependents'] = feat_eng_cust['kidhome'] + feat_eng_cust['teenhome']

        # Criando a feature Age.
        feat_eng_cust['age'] = 2023 - feat_eng_cust['year_birth']

        # RFM

        # Criando a feature total de compras
        feat_eng_cust['total_purchases'] = feat_eng_cust['numcatalogpurchases'] + feat_eng_cust['numdealspurchases'] + feat_eng_cust['numstorepurchases'] + feat_eng_cust['numwebpurchases']

        #  Criando a feature tempo de relecionamento
        current_date = datetime.today()
        feat_eng_cust['relationship_duration'] = (current_date.year - feat_eng_cust['dt_customer'].dt.year) 

        # Criando a feature Frequencia
        feat_eng_cust['frequency'] = feat_eng_cust['total_purchases'] / feat_eng_cust['relationship_duration']
        
        # Criando a feature monetario.
        feat_eng_cust['monetary'] = feat_eng_cust['fish'] + feat_eng_cust['fruits'] + feat_eng_cust['gold'] + feat_eng_cust['meat'] + feat_eng_cust['sweets'] + feat_eng_cust['wines']

        # Criando a feature average purchase value.
        feat_eng_cust['avg_purchase_value'] = feat_eng_cust['monetary'] / feat_eng_cust['total_purchases'].replace(0, np.nan)

        # Dropando outlier
        feat_eng_cust.drop(feat_eng_cust.loc[feat_eng_cust['avg_purchase_value'] > 1500].index, inplace=True)
        
        # Obtaining the ID column for further use in customer segmentation.
        ids = feat_eng_cust['id']

        # Dropping irrelevant columns.
        feat_eng_cust.drop(columns=['z_costcontact', 'z_revenue', 'id', 'kidhome', 
                         'teenhome', 'complain', 'response', 
                        'acceptedcmp1', 'acceptedcmp2', 'acceptedcmp3', 
                        'acceptedcmp4', 'acceptedcmp5', 'dt_customer',
                        'year_birth', 'total_purchases'], inplace=True)

        return feat_eng_cust, ids

    except Exception as e:
        raise CustomException(e, sys)





def silhouette_analysis(data_pca, n_clusters_range=range(2, 11)):
   
    silhouette_scores = {}
    ncols = 2  # Número de gráficos por linha
    n_clusters = len(n_clusters_range)  # Número total de gráficos
    nrows = (n_clusters + ncols - 1) // ncols  # Calcula o número de linhas necessárias
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5), constrained_layout=True)
    axs = axs.flatten()  # Achata o array de eixos para facilitar o acesso

    for idx, k in enumerate(n_clusters_range):
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data_pca)
        
        # Calculando o Silhouette Score médio
        silhouette_avg = silhouette_score(data_pca, clusters)
        silhouette_scores[k] = silhouette_avg
        
        
        # Análise de silhueta
        ax = axs[idx]
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(data_pca) + (k + 1) * 10])

        sample_silhouette_values = silhouette_samples(data_pca, clusters)
        y_lower = 10
        
        for i in range(k):
            # Valores do Silhouette para cada cluster
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax.set_title(f"Silhueta para {k} clusters")
        ax.set_xlabel("Valores do coeficiente de Silhueta")
        ax.set_ylabel("Cluster")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="Média Silhueta")
        ax.legend()

    # Ocultar os gráficos não usados (se houver)
    for ax in axs[n_clusters:]:
        ax.axis("off")

    plt.show()
    
    return silhouette_scores


