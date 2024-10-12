# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Constants and Configurations
plt.rcParams.update({'font.size': 14})  # Change this value as needed
PATH = "/home/filip/Dokumenty/web scraping samochody/zhtml/samochody05082024pythonCH/wersja 3/analiza/"
REFERENCE_DATE = pd.to_datetime('2024-08-08')
#for generating plots for Switzerland change this to CH
COUNTRY='PL'
#COUNTRY='CH'
PLOT_DIR = 'wykresy'+'_'+COUNTRY+'/'
if COUNTRY=='PL':
    KLASY_Najp2024  = "klasy100najp2024PL.csv"    
elif COUNTRY=='CH':
    KLASY_Najp2024  = "klasy150najp2024CH.csv"
DANE_SAMOCHODOW = "czysteDaneSamochodowWybraneModele"+COUNTRY+".csv"



# Function Definitions

def load_data(file_path, sep=";", decimal=","):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, sep=sep, decimal=decimal)

def prepare_data(df, ref_date, Klasy100najp):
    """Prepare the dataframe by merging, converting dates, and calculating the age of offers. ref_date is a date of downloading offers; Klasy are car types; we assume that df is already with unique rows"""
    # Merge with class information
    Klasy100najp.rename(columns={'Model': 'model', 'Class': 'klasa'}, inplace=True)
    merged = df.merge(Klasy100najp[['model', 'klasa']], on='model', how='left')
    merged['klasa'] = merged['klasa'].fillna('X')
    # Convert dates and calculate the age of offers
    merged['cena'] = (merged['cena'] / 1000)  # Convert 'cena' to thousands
    if COUNTRY=='CH':
        merged['utworzono'] = pd.to_datetime(merged['utworzono'])
        merged['wiek_ogloszenia'] = (ref_date - merged['utworzono']).dt.days
        merged = merged[['id', 'model', 'klasa', 'cena', 'rok', 'przebieg', 'pojemnosc_silnika', 'skrzynia_biegow', 'wiek_ogloszenia', 'link']]  # Select specific columns
        merged['przebieg'] = (merged['przebieg'] / 1000)  # Convert to integer
    elif COUNTRY=='PL':
        merged['cena'] = (merged['cena'] / 4.5)  # Convert 'cena' to CHF
        merged = merged[['id', 'model', 'klasa', 'cena', 'rok', 'przebieg', 'pojemnosc_silnika', 'skrzynia_biegow', 'link']]  # Select specific columns
    return merged

def setup_visual_style():
    plt.style.use('ggplot')
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': 'lightgray',
        'axes.grid': True,
        'grid.alpha': 0.8,
        'grid.linestyle': '--',
        'grid.color': 'gray',
        'figure.figsize': (12, 8)
    })
    sns.set_context("talk")
    sns.set_style("whitegrid", {
        'axes.facecolor': '#F9F9F9',
        'grid.color': '.85',
        'grid.linestyle': '-'
    })
# Function to filter data based on mileage (przebieg)
def filter_data_by_mileage(df, min_mileage, max_mileage):
    return df[(df['przebieg'] > min_mileage) & (df['przebieg'] < max_mileage)]


# Produce a grid plot with histograms
def grid_plot(filtered_data, nazwa_pliku='RozkladPrzebieguStare25PodzialnaModele'+COUNTRY+'.svg', os1='przebieg', os1_nazwa='Przebieg (w tys. km)', plot_title="", os1_min=50, os1_max=300, os1_step=10):
    # Create a color palette based on the unique models
    unique_models = sorted(filtered_data['model'].unique())
    colors = sns.color_palette("hls", len(unique_models))  # Using a color palette with distinct colors
    color_dict = dict(zip(unique_models, colors))  # Create a mapping from model to color
    # Create a FacetGrid for plotting
    g = sns.FacetGrid(filtered_data, col='model', col_order=unique_models, col_wrap=5, height=4, sharex=True, sharey=True)
    # Map the histogram to the grid
    g.map(sns.histplot, os1, bins=np.arange(os1_min, os1_max + os1_step, os1_step), edgecolor='gray')
    # Map the histogram to the grid using the hue parameter
    g.map_dataframe(sns.histplot, 
                    x=os1, 
                    bins=np.arange(os1_min, os1_max + os1_step, os1_step), 
                    edgecolor='gray', 
                    hue='model',  # Use 'model' to color bars based on the model
                    palette=color_dict)  # Set the color palette
    # Set titles and labels
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(os1_nazwa, "Liczba samochodów")
    g.set(xlim=(os1_min, os1_max))
    for ax in g.axes.flatten():
        ax.set_facecolor('#FAFAFA')  # Set the background color to light gray
        ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)  # Grid on the y-axis
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  # Grid on the x-axis
    # Show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top to make space for the title
    g.fig.suptitle(plot_title, fontsize=20)
    # Save the plot as a svg file
    plt.savefig(f'{PLOT_DIR}{nazwa_pliku}', format='svg', dpi=300)

# Produce a grid plot with scatter plots between 'przebieg' and 'cena'
def scatter_plot(filtered_data, nazwa_pliku='ScatterPlotPrzebiegCena'+COUNTRY+'.svg', os1='przebieg', os2='cena', x_min=None, y_min=None, x_max=None, y_max=None, x_dist=20, y_dist=20, os1_nazwa='Przebieg (w tys. km)', os2_nazwa='Cena (w '+COUNTRY+')', plot_title="Scatter Plot of Przebieg vs Cena"):
    # Calculate x_min, x_max, y_min, y_max if not provided
    x_min = filtered_data[os1].min().round(0).astype(int) if x_min is None else x_min
    x_max = filtered_data[os1].max().round(0).astype(int) if x_max is None else x_max
    y_min = filtered_data[os2].min().round(0).astype(int) if y_min is None else y_min
    y_max = filtered_data[os2].max().round(0).astype(int) if y_max is None else y_max
    # Create a color palette based on the unique models
    unique_models = sorted(filtered_data['model'].unique())
    colors = sns.color_palette("hls", len(unique_models))  # Using a color palette with distinct colors
    color_dict = dict(zip(unique_models, colors))  # Create a mapping from model to color
    # Create a FacetGrid for plotting
    g = sns.FacetGrid(filtered_data, col='model', col_order=unique_models, col_wrap=5, height=4, sharex=True, sharey=True)
    # Map the scatterplot to the grid
    g.map_dataframe(sns.scatterplot, 
                    x=os1, 
                    y=os2, 
                    hue='model',  # Use 'model' to color points based on the model
                    palette=color_dict, 
                    alpha=0.4)  # Adjust alpha for better visibility
    # Set titles and labels
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels(os1_nazwa, os2_nazwa)
    # Add regression line for each subplot
    for ax in g.axes.flatten():
        # Get data for the current subplot
        model_data = filtered_data[filtered_data['model'] == ax.get_title()]
        x = model_data[os1]
        y = model_data[os2]
        if len(x) > 1:  # Ensure there are enough points to fit a line
            # Fit a regression line
            coeffs = np.polyfit(x, y, 3)  # Polynomial fit with degree 3
            poly_eq = np.poly1d(coeffs)
            # Create x values for the regression line
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = poly_eq(x_fit)
            # Plot the regression line
            ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Regression Line')
#        # Set the y-axis limit for each subplot
#        ax.set_ylim(y_min, y_max)
#        # Set the x-axis limit for each subplot
#        ax.set_xlim(x_min, x_max)
        # Add a light gray background to each subplot
        ax.set_facecolor('#FAFAFA')  # Set the background color to light gray
        ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.7)  # Grid on the y-axis
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.7)  # Grid on the x-axis
        # Rotate x-ticks by 90 degrees
        ax.tick_params(axis='x', rotation=90)
        
        # Set x-ticks and reverse them if the x-axis is 'rok'
        if os1 == 'rok':
            x_ticks = np.arange(x_max, x_min - x_dist, -x_dist)  # Generate reversed x-ticks
        else:
            x_ticks = np.arange(x_min, x_max + x_dist, x_dist)  # Generate x-ticks at a distance of x_dist
        ax.set_xticks(x_ticks)
        ax.set(xlim=(x_min, x_max))


        # Set y-ticks and reverse them if the y-axis is 'rok'
        if os2 == 'rok':
            y_ticks = np.arange(y_max, y_min - y_dist, -y_dist)  # Generate reversed y-ticks
        else:
            y_ticks = np.arange(y_min, y_max + y_dist, y_dist)  # Generate y-ticks at a distance of y_dist
        ax.set_yticks(y_ticks)
        ax.set(ylim=(y_min, y_max))

        # Reverse the axis if needed
        if os1 == 'rok':
            ax.invert_xaxis()  # Invert the x-axis if 'rok'
        if os2 == 'rok':
            ax.invert_yaxis()  # Invert the y-axis if 'rok'
    # Save the plot as a svg file
    g.fig.suptitle(plot_title, fontsize=20)
    #g.add_legend()
    g.tight_layout()
    g.savefig(f'{PLOT_DIR}{nazwa_pliku}', format='svg', dpi=300)
    
def plot_histogram(df, column, title, xlabel, ylabel, bins, xticks, output_file, color='lightgreen', mean_color='red', median_color='blue'):
    """Plot a histogram with mean and median lines."""
    plt.figure(figsize=(10, 8))
    sns.histplot(df[column], bins=bins, color=color, edgecolor='gray')
    plt.axvline(x=df[column].mean(), color=mean_color, linewidth=1, label='Mean')
    plt.axvline(x=df[column].median(), color=median_color, linewidth=1, label='Median')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks, rotation=90)
    plt.legend()
    plt.savefig(f'{PLOT_DIR}{output_file}', format='svg', dpi=300)
    plt.close()
    

def plot_barh(df, models, title, xlabel, ylabel, output_file, figsize_x=20, figsize_y=10):
    """Plot a horizontal bar chart for the most popular models."""
    plt.figure(figsize=(figsize_x, figsize_y))
    # Filter the dataframe to include only the selected models
    filtered_df = df[df['model'].isin(models)]
    # Get the count of each model and sort them
    model_counts = filtered_df['model'].value_counts().sort_values()
    # Map each model to its corresponding 'klasa'
    model_klasa = filtered_df.drop_duplicates('model').set_index('model')['klasa']
    # Create a color palette based on unique classes in 'klasa'
    unique_classes = model_klasa.unique()
    class_palette = sns.color_palette("hls", len(unique_classes))
    class_color_dict = dict(zip(unique_classes, class_palette))
    # Map colors to the models based on their 'klasa'
    colors = model_klasa.map(class_color_dict)
    # Plot the horizontal bar chart with colors based on 'klasa'
    model_counts.plot(kind='barh', color=colors.loc[model_counts.index], edgecolor='black')
    # Set the titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}{output_file}', format='svg', dpi=300)
    plt.close()


def perform_linear_regression(df, x_col, y_col):
    """Perform linear regression and return the model."""
    lin_reg = LinearRegression()
    X = df[[x_col]]
    y = df[y_col]
    lin_reg.fit(X, y)
    return lin_reg, X, y

def perform_polynomial_regression(df, x_col, y_col, degree=3):
    """Perform polynomial regression and return the model, X, and y."""
    X = df[[x_col]]
    y = df[y_col]
    # Create a pipeline to transform X to polynomial features and fit a linear model
    polynomial_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_model.fit(X, y)
    return polynomial_model, X, y

    
def plot_regression(df, X, y, poly_model, title, xlabel, ylabel, output_file):
    """Plot scatter and polynomial regression line."""
    plt.figure(figsize=(12, 6))
    # Scatter plot of actual data points
    sns.scatterplot(x=X.squeeze(), y=y, data=df, color='blue', alpha=0.5, label='Actual Data')
    # Generate a smooth curve for the polynomial regression line
    X_sorted = np.sort(X.squeeze()).reshape(-1, 1)  # Sort X and reshape for prediction
    # Convert sorted X to a DataFrame with column name for compatibility
    X_sorted_df = pd.DataFrame(X_sorted, columns=[X.columns[0]])
    y_pred = poly_model.predict(X_sorted_df)
    # Plot the polynomial regression line
    sns.lineplot(x=X_sorted.squeeze(), y=y_pred, color='red', label='Polynomial Regression Line')
    # Add title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    y_max = min(460, y.max().round(0).astype(int))
    plt.yticks(range(0, y_max, 20))
    plt.ylim(0,y_max)
    plt.legend()
    plt.tight_layout()
    # Save the plot to a file
    plt.savefig(f'{PLOT_DIR}{output_file}', format='svg', dpi=300)
    plt.close()
    

# Function to standardize the data before clustering
def standardize_data(df, features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    return pd.DataFrame(scaled_features, columns=features)

# KMeans clustering
def kmeans_clustering(df, features, n_clusters=5):
    df_scaled = standardize_data(df, features)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    return kmeans.cluster_centers_, df

# Function to plot Współczynnik utraty wartości samochodu
def wsp_plot(DF50Nowe, nazwa_pliku='ScatterPlotWspUtratyWart'+COUNTRY+'.svg', os1='rok', os2='cena', plot_title="Współczynnik utraty wartości samochodu z wiekiem >1999"):
    df_copy = DF50Nowe.copy()
    if os1 == 'rok':
        df_copy[os1] = 2024 - df_copy[os1]
    if os2 == 'rok':
        df_copy[os2] = 2024 - df_copy[os2]
    reg_cena = df_copy.groupby('model').apply(lambda x: LinearRegression().fit(x[[os1]], x[os2])).reset_index()
    reg_cena['Slope_cena'] = reg_cena[0].apply(lambda model: model.coef_[0])
    reg_cena = reg_cena[['model', 'Slope_cena']]
    reg_cena = reg_cena.sort_values(by='Slope_cena')

    plt.figure(figsize=(20, 10))
    sns.scatterplot(x='Slope_cena', y='model', data=reg_cena, color='green', s=100)
    plt.title(plot_title)
    plt.xlabel("Współczynnik")
    plt.ylabel("")
    plt.gca().set_facecolor('#FAFAFA')
    plt.grid(color='gray', linestyle='--', linewidth=0.7)
    x_min, x_max = plt.xlim()
    plt.xticks(np.arange(np.floor(x_min), np.ceil(x_max), 1))
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}{nazwa_pliku}", format='svg', dpi=300)

# Function to plot elipsy
def elipsy(DF50, Klasynajp, nazwa='Elipsy50'+COUNTRY):
    exclude_models = ['Volkswagen T-Roc', 'Mazda CX-5']
    DF50NowePrzebieg2000 = DF50[~DF50['model'].isin(exclude_models)]
    reg_cena = (DF50NowePrzebieg2000[['model', 'rok', 'cena']]
                .groupby('model')
                .apply(lambda x: sm.OLS(x['cena'], sm.add_constant(x['rok'])).fit().params[1])
                .reset_index(name='Slope_cena'))
    reg_przebieg = (DF50NowePrzebieg2000[['model', 'rok', 'przebieg']]
                    .groupby('model')
                    .apply(lambda x: sm.OLS(x['przebieg'], sm.add_constant(x['rok'])).fit().params[1])
                    .reset_index(name='Slope_przebieg'))
    reg_both = pd.merge(reg_przebieg, reg_cena, on='model')
    
    scaler = StandardScaler()
    X_both = scaler.fit_transform(reg_both[['Slope_przebieg', 'Slope_cena']])
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X_both)
    reg_both['klasa'] = kmeans.labels_
    reg_both = reg_both.merge(Klasynajp, left_on='model', right_on='model', how='left')

    plt.figure(figsize=(14, 20))
    sns.scatterplot(data=reg_both, x='Slope_cena', y='Slope_przebieg', hue='klasa_y', palette='hls', legend=False)
    
    for cluster in reg_both['klasa_x'].unique():
        cluster_data = reg_both[reg_both['klasa_x'] == cluster]
        if len(cluster_data) > 1:
            cov = np.cov(cluster_data[['Slope_cena', 'Slope_przebieg']], rowvar=False)
            mean = cluster_data[['Slope_cena', 'Slope_przebieg']].mean().values
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            theta = np.arctan(u[1] / u[0])
            ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], angle=np.degrees(theta), color='gray', alpha=0.2)
            plt.gca().add_patch(ell)
    
    for i in range(len(reg_both)):
        plt.text(reg_both['Slope_cena'].iloc[i], reg_both['Slope_przebieg'].iloc[i], 
                 reg_both['model'].iloc[i], fontsize=12)
    
    plt.title("Współczynnik eksploatacji vs współczynnik utraty wartości")
    plt.xlabel("Współczynnik utraty wartości\n dalej od zera = gorzej")
    plt.ylabel("Współczynnik eksploatacji\n dalej od zera = lepiej")
    plt.axis('equal')
    plt.savefig(f'{PLOT_DIR}{nazwa}.svg', format='svg', dpi=300)



def graph_cars(DF, car_names, os_x='przebieg', os_y='cena', x_min=50, x_max=350, y_max=50, y_dist=20):
    car_names_sorted = sorted(car_names)
    filtered_df = DF[
        (DF['model'].isin(car_names_sorted)) &
        (DF[os_x] > x_min) & 
        (DF[os_x] < x_max)
    ]
    color_palette = sns.color_palette("hls", len(car_names_sorted))
    color_dict = dict(zip(car_names_sorted, color_palette))
    plt.figure(figsize=(20, 12))
    ax = plt.gca()
    ax.set_facecolor('#FAFAFA')
    sns.scatterplot(data=filtered_df, x=os_x, y=os_y, hue='model', palette=color_dict, s=20, alpha=0.5, legend='full')
    for model in car_names_sorted:
        subset = filtered_df[filtered_df['model'] == model]
        sns.regplot(data=subset, x=os_x, y=os_y, scatter=False, color=color_dict[model], label=model, order=4)
    plt.title(os_y + " w zależności od " + os_x + " auta, w podziale na marki")
    plt.xlabel(os_x + " w tys.")
    plt.ylabel(os_y + " w tys. "+COUNTRY+'')
    
#    # Set x-ticks and reverse them if the x-axis is 'rok'
#    if os_x == 'rok':
#        plt.xticks = np.arange(x_max, x_min - 5, -5)  # Generate reversed x-ticks
#    else:
#        plt.xticks = np.arange(x_min, x_max + 5, 5)  # Generate x-ticks at a distance of x_dist
    plt.ylim(0, y_max)
    if os_x!='rok':
        plt.xlim(0, x_max)
    plt.yticks(range(0, y_max, y_dist))
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(title='Model')
    # Extract the second part of each car name and join them with underscores
    name_parts = ["_".join(model.split()[1:]) for model in car_names_sorted]
    name = "_".join(name_parts) + "_" + os_x + "_" + os_y + "_"+COUNTRY+".svg"
    plt.savefig(f'{PLOT_DIR}{name}.svg', format='svg', dpi=300)
    
#function to find the most attractive offers    
def okazje(DF, nazwa_csv="okazje", cena_pr_min=-50, cena_pr_max=-5, przebieg_pr_min=-100, przebieg_pr_max=-5):
# Step 3: Create the 'okazje' DataFrame
    
# Assuming DF50NowePrzebieg2000 and DF100 are your DataFrames
# Step 1: Calculate averages
    auta_srednie = (DF
                    .groupby(['model', 'rok'])
                    .agg(przebieg_sredni=('przebieg', 'mean'),
                         cena_srednia=('cena', 'mean'),
                         n=('przebieg', 'size'))
                    .reset_index())
    # Round the columns to two decimal places
    auta_srednie['przebieg_sredni'] = auta_srednie['przebieg_sredni'].round(2)
    auta_srednie['cena_srednia'] = auta_srednie['cena_srednia'].round(2)
    
    if COUNTRY=='CH':
        okazje = (DF
              .merge(auta_srednie, on=['model', 'rok'], suffixes=('', '_srednia'))
              .assign(cena_procent=lambda x: round(100 * (x['cena'] / x['cena_srednia'] - 1), 1),
                      przebieg_procent=lambda x: round(100 * (x['przebieg'] / x['przebieg_sredni'] - 1), 1))
              .query(f'{cena_pr_min} <= cena_procent <= {cena_pr_max} and {przebieg_pr_min} <= przebieg_procent <= {przebieg_pr_max}')
              .sort_values(by=['model', 'cena_procent', 'przebieg_procent'], ascending=[True, False, False])
              .loc[:, ['model', 'rok', 'cena', 'cena_srednia', 'cena_procent', 'przebieg', 'przebieg_sredni', 'przebieg_procent', 'wiek_ogloszenia', 'link']])
    elif COUNTRY=='PL':
        okazje = (DF
              .merge(auta_srednie, on=['model', 'rok'], suffixes=('', '_srednia'))
              .assign(cena_procent=lambda x: round(100 * (x['cena'] / x['cena_srednia'] - 1), 1),
                      przebieg_procent=lambda x: round(100 * (x['przebieg'] / x['przebieg_sredni'] - 1), 1))
              .query(f'{cena_pr_min} <= cena_procent <= {cena_pr_max} and {przebieg_pr_min} <= przebieg_procent <= {przebieg_pr_max}')
              .sort_values(by=['model', 'cena_procent', 'przebieg_procent'], ascending=[True, False, False])
              .loc[:, ['model', 'rok', 'cena', 'cena_srednia', 'cena_procent', 'przebieg', 'przebieg_sredni', 'przebieg_procent', 'link']])
                     
    # Step 4: View the 'okazje' DataFrame
#    print(okazje)
    okazje.to_csv(PLOT_DIR+nazwa_csv+'.csv', sep=';', index=False)
    return okazje


# Main Logic

def main():
    # Set working directory
    os.chdir(PATH)
    # Check if the directory exists, if not, create it
    if not os.path.exists(PATH + PLOT_DIR):
        os.makedirs(PATH + PLOT_DIR)
    
    # Load data
    RamkaDanych = load_data(DANE_SAMOCHODOW)
    Klasy100najp = load_data(KLASY_Najp2024)
    
#    if RamkaDanych.empty or Klasy100najp.empty:
#        print("Error loading data. Exiting.")
#        return
    
    # Prepare data
    RamkaDanych = prepare_data(RamkaDanych, REFERENCE_DATE, Klasy100najp)
    ######ONLY NEW OFFERS
    #RamkaDanych = RamkaDanych[RamkaDanych['wiek_ogloszenia']<90]
    
        # Favorite models
    favourite_models = ["Mercedes-Benz KlasaA", "Mercedes-Benz KlasaB", "Mercedes-Benz KlasaC", "Seat Leon", "Renault Captur", "Hyundai I30", "Toyota Corolla", "Volkswagen Golf", "Volkswagen Passat", "Volkswagen Tiguan", "Renault Megane", "Renault Scenic", "Alfa-Romeo Giulietta", "Opel Mokka", "Skoda Octavia", "Audi A4", "Ford Kuga", "Skoda Octavia"]
    DF_favourite = RamkaDanych[RamkaDanych['model'].isin(favourite_models)]
#    DF_favourite=DF_favourite[DF_favourite['rok']>2005]
#    RamkaDanych=DF_favourite
    
    # Popular models
    najpopularniejsze25 = RamkaDanych['model'].value_counts().nlargest(25).index.tolist()
    najpopularniejsze50 = RamkaDanych['model'].value_counts().nlargest(50).index.tolist()
    najpopularniejsze100 = RamkaDanych['model'].value_counts().nlargest(100).index.tolist()
    # Filter DataFrames based on model popularity
    DF100 = RamkaDanych[RamkaDanych['model'].isin(najpopularniejsze100)]
    DF50 = RamkaDanych[RamkaDanych['model'].isin(najpopularniejsze50)]
    DF25 = RamkaDanych[RamkaDanych['model'].isin(najpopularniejsze25)]

    
    # 1. Set up the visual style
    setup_visual_style()
    
    # Count occurrences of each model
    model_counts = RamkaDanych.value_counts(subset=['model', 'klasa']).reset_index(name='counts')
    model_counts.to_csv(PLOT_DIR+'Popularnosc_modeli'+COUNTRY+'.csv', sep=';', index=False)
    

    
    # Plotting
    plot_barh(RamkaDanych, najpopularniejsze50, "Liczba oferowanych samochodów według marki", "Liczba ofert", "Model", "najpopularniejsze50"+COUNTRY+".svg", figsize_x=20, figsize_y=12)
    plot_barh(RamkaDanych, najpopularniejsze100, "Liczba oferowanych samochodów według marki", "Liczba ofert", "Model", "najpopularniejsze100"+COUNTRY+".svg", figsize_x=20, figsize_y=22)
    
    # Linear regression and plotting
    lin_reg, X, y = perform_linear_regression(RamkaDanych, 'rok', 'przebieg')
    poly_model, X, y = perform_polynomial_regression(RamkaDanych, 'rok', 'przebieg', degree=3)
    plot_regression(RamkaDanych, X, y, poly_model, "Zależność przebiegu od roku produkcji samochodu", "Rok produkcji", "Przebieg (w tys. km)", "regression_plot.svg")


    
    # Additional histogram plots
    if COUNTRY=='CH':
        plot_histogram(RamkaDanych, 'wiek_ogloszenia', "Rozkład wieku ogłoszeń", "Wiek w dniach", "Liczba samochodów", np.arange(0, 700, 30), np.arange(0, 700, 30), "RozkładWiekuOgloszen"+COUNTRY+".svg")
        plot_histogram(DF50, 'wiek_ogloszenia', "Rozkład wieku ogłoszeń 50 najp. sam.", "Wiek w dniach", "Liczba samochodów", np.arange(0, 200, 7), np.arange(0, 200, 7), "RozkładWiekuWszystkichOferowanychNowychOgloszen_DF50_"+COUNTRY+".svg")
        
    plot_histogram(RamkaDanych, 'przebieg', "Rozkład przebiegu", "Przebieg (w tys. km)", "Liczba samochodów", np.arange(0, 350, 10), np.arange(0, 350, 10), "RozkładPrzebiegu"+COUNTRY+".svg")
    plot_histogram(DF50, 'przebieg', "Rozkład przebiegu  50 najp. sam.", "Przebieg (w tys. km)", "Liczba samochodów", np.arange(0, 350, 10), np.arange(0, 350, 10), "RozkładPrzebiegu_DF50_"+COUNTRY+".svg")

#    # Additional regression and plotting for DF50
#    lin_reg_DF50, X_DF50, y_DF50 = perform_linear_regression(DF50, 'rok', 'przebieg')
#    plot_regression(DF50, X_DF50, y_DF50, "Zależność przebiegu od roku produkcji samochodu (DF50)", "Rok produkcji", "Przebieg (w tys. km)", "regression_DF50_plot.svg")
    

    
    # 2. Filter the data by mileage range
    filtered_data = filter_data_by_mileage(DF25, 50, 300)
    
    # 3. Generate a grid plot with histograms
    grid_plot(filtered_data=filtered_data, nazwa_pliku='RozkladPrzebieguStare25PodzialnaModele'+COUNTRY+'.svg', plot_title="Rozkład przebiegu auta, w podziale na marki")
    if COUNTRY=='CH':
        grid_plot(filtered_data=DF25, nazwa_pliku='RozkladWiekuOgloszenStare25PodzialnaModele'+COUNTRY+'.svg', os1_nazwa='Wiek ogłoszenia (30 d. przed.)', plot_title="Rozkład wieku ogłoszeń, w podziale na marki", os1_min=0, os1_max=400, os1_step=7)
    
    # 4. Generate a grid plot with scatter plots (mileage vs price)
    scatter_plot(filtered_data, nazwa_pliku='ScatterPlotPrzebiegCena'+COUNTRY+'.svg', plot_title="Scatter Plot of Przebieg vs Cena", y_min=0, y_max=60, y_dist=10,x_dist=25)
    scatter_plot(filtered_data, nazwa_pliku='ScatterPlotRokPrzebieg'+COUNTRY+'.svg', plot_title="Scatter Plot of Rok vs Przebieg", os2='przebieg', os1='rok', x_min=1995, y_min=0, y_max=None, x_max=None, y_dist=25, x_dist=5, os1_nazwa='Rok', os2_nazwa='Przebieg (w tys. km)')
    scatter_plot(filtered_data, nazwa_pliku='ScatterPlotCenaRok'+COUNTRY+'.svg', plot_title="Scatter Plot of Cena vs Rok", os1='rok', os2='cena', x_min=1995, y_min=0, x_max=None, y_max=60, x_dist=5, y_dist=10, os1_nazwa='Rok', os2_nazwa='Cena (w '+COUNTRY+')')
    
    
    # Generate and save the wsp plots
    wsp_plot(filtered_data, nazwa_pliku='ScatterPlotWspUtratyWartzWiekiem.svg', os1='rok', os2='cena', plot_title="Współczynnik utraty wartości samochodu z wiekiem >1999")
    wsp_plot(filtered_data, nazwa_pliku='ScatterPlotWspUtratyWartzPrzebiegiem.svg', os1='przebieg', os2='cena', plot_title="Współczynnik utraty wartości samochodu z przebiegiem >1999")

    # Generate and save the elipsy plots
    elipsy(DF50, Klasynajp=Klasy100najp, nazwa='Elipsy50'+COUNTRY)
    elipsy(DF100, Klasynajp=Klasy100najp, nazwa='Elipsy100'+COUNTRY)

    graph_cars(DF=RamkaDanych, car_names=["Seat Leon", "Hyundai I30", "Volkswagen Golf"], x_min=50, x_max=350, os_y='cena', y_max=40, y_dist=2)
    graph_cars(DF=RamkaDanych, car_names=["Mercedes-Benz KlasaA", "Mercedes-Benz KlasaB", "Mercedes-Benz KlasaC", "Seat Leon", "Renault Captur", "Hyundai I30", "Toyota Corolla", "Volkswagen Golf", "Volkswagen Passat", "Volkswagen Tiguan", "Renault Megane", "Renault Scenic", "Alfa-Romeo Giulietta", "Opel Mokka", "Skoda Octavia", "Audi A4"], x_min=50, x_max=350, os_y='cena', y_max=40, y_dist=2)
    graph_cars(DF=RamkaDanych, car_names=["Mercedes-Benz KlasaA", "Mercedes-Benz KlasaB", "Mercedes-Benz KlasaC", "Seat Leon", "Renault Captur", "Hyundai I30", "Toyota Corolla"], x_min=50, x_max=350, os_y='cena', y_max=40, y_dist=2)
    graph_cars(DF=RamkaDanych, car_names=["Volkswagen Golf", "Volkswagen Passat", "Volkswagen Tiguan", "Renault Megane", "Renault Scenic", "Alfa-Romeo Giulietta", "Opel Mokka", "Skoda Octavia", "Audi A4"], x_min=50, x_max=350, os_y='cena', y_max=40, y_dist=2)
    graph_cars(DF=RamkaDanych, car_names=["Mercedes-Benz KlasaB", "Seat Leon", "Renault Captur", "Hyundai I30", "Volkswagen Golf", "Renault Megane", "Renault Scenic", "Skoda Octavia", "Ford Kuga"], x_min=50, x_max=350, os_y='cena', y_max=40, y_dist=2)
    graph_cars(DF=RamkaDanych, car_names=["Seat Leon", "Volkswagen Golf", "Renault Megane", "Skoda Octavia", "Ford Kuga"], x_min=50, x_max=350, os_y='cena', y_max=40, y_dist=2)
    graph_cars(DF=RamkaDanych,car_names=["Mercedes-Benz KlasaB", "Seat Leon", "Hyundai I30", "Volkswagen Golf", "Volkswagen Golf Sportsvan", "Renault Megane", "Renault Scenic", "Skoda Octavia", "Ford Kuga"], os_x='przebieg', os_y='cena', x_min=50, x_max=350, y_max=40, y_dist=2)
 
    graph_cars(DF=RamkaDanych, car_names=["Seat Leon", "Volkswagen Golf", "Renault Megane", "Skoda Octavia", "Ford Kuga"], os_x='rok', os_y='przebieg', x_min=2005, x_max=2023, y_max=350)
    graph_cars(DF=RamkaDanych,car_names=["Mercedes-Benz KlasaB", "Seat Leon", "Hyundai I30", "Volkswagen Golf", "Volkswagen Golf Sportsvan", "Renault Megane", "Renault Scenic", "Skoda Octavia", "Ford Kuga"], os_x='rok', os_y='przebieg', x_min=2005, x_max=2023, y_max=350)
    graph_cars(DF=RamkaDanych,car_names=["Mercedes-Benz KlasaB", "Seat Leon", "Hyundai I30", "Volkswagen Golf", "Volkswagen Golf Sportsvan", "Renault Megane", "Renault Scenic", "Skoda Octavia", "Ford Kuga"], os_x='rok', os_y='cena', x_min=2005, x_max=2023, y_max=40, y_dist=2)
        
    RamkaDanychOkazje2= (RamkaDanych
              .query('(2015 <= rok <= 2023) and (5 <= cena <= 12) and (100< przebieg < 160)'))
    # Count occurrences of each model
    model_counts_tanie2 = RamkaDanychOkazje2.value_counts(subset=['model', 'klasa']).reset_index(name='counts')
    model_counts_tanie2.to_csv(PLOT_DIR+'Popularnosc_Btanich_modeli'+COUNTRY+'.csv', sep=';', index=False)
    najpopularniejsze125 = RamkaDanychOkazje2['model'].value_counts().nlargest(125).index.tolist()
    plot_barh(RamkaDanychOkazje2, najpopularniejsze125, "Liczba oferowanych samochodów według marki 15<r<23, 5<c<12, 100<p<160", "Liczba ofert", "Model", "najpopularniejsze125Tanie"+COUNTRY+".svg", figsize_x=20, figsize_y=22)
    
    RamkaDanychOkazje2= (RamkaDanych
              .query('(2012 <= rok <= 2023) and (5 <= cena <= 13) and (100< przebieg < 160)'))
    # Count occurrences of each model
    model_counts_tanie2 = RamkaDanychOkazje2.value_counts(subset=['model', 'klasa']).reset_index(name='counts')
    model_counts_tanie2.to_csv(PLOT_DIR+'Popularnosc_Tanich_modeli'+COUNTRY+'.csv', sep=';', index=False)
    najpopularniejsze125 = RamkaDanychOkazje2['model'].value_counts().nlargest(125).index.tolist()
    plot_barh(RamkaDanychOkazje2, najpopularniejsze125, "Liczba oferowanych samochodów według marki 12<r<23, 5<c<13, 100<p<160", "Liczba ofert", "Model", "najpopularniejsze125Btanie"+COUNTRY+".svg", figsize_x=20, figsize_y=22)
        
    okazjeDF=okazje(RamkaDanych, nazwa_csv="okazje")
    if COUNTRY=='CH':
        plot_histogram(okazjeDF, 'wiek_ogloszenia', "Rozkład wieku ogłoszenia wśród okazji", "Wiek ogłoszenia", "Liczba okazji", np.arange(0, 350, 7), np.arange(0, 350, 7), "RozkładOkazji_"+COUNTRY+".svg")
        plot_histogram(okazjeDF[okazjeDF['model'].isin(["Skoda Octavia"])], 'wiek_ogloszenia', "Rozkład wieku ogłoszenia wśród okazji", "Wiek ogłoszenia", "Liczba okazji", np.arange(0, 350, 7), np.arange(0, 350, 7), "RozkładOkazji_"+COUNTRY+".svg")

    RamkaDanychBWysluzone= (RamkaDanych.query('(300< przebieg)'))
    RamkaDanychBWysluzone.to_csv(PLOT_DIR+'B_Wysluzone'+COUNTRY+'.csv', sep=';', index=False)
# Entry Point
if __name__ == "__main__":
    main()
