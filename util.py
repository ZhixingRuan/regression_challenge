import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set()


# change data to num
class Str2NumTransform:
    
    '''
    Preprocess the original data
    Transform the 'cost', 'price', 'weight', 'height', 'width', 'depth' to numerical values
    Use transform(df), output a new dataframe
    '''
    
    def transform(self, df):
        df['price'] = df['price'].map(self.price2num)
        df['cost'] = df['cost'].map(self.cost2num)
        df['weight'] = df['weight'].map(self.weight2num)
        df['height'] = df['height'].map(self.height2num)
        df['width'] = df['width'].map(self.wd2num)
        df['depth'] = df['depth'].map(self.wd2num)
        return df
    
    def price2num(self, x):
        if type(x) == str:
            x = x.strip('$').replace(',', '')
            return float(x)
    
    def cost2num(self, x):
        if type(x) == str:
            x = x.strip('$').strip('k')
            return float(x)*1000
    
    def weight2num(self, x):
        if type(x) == str:
            x = x.strip('Kg').split('Ton')
            return float(x[0])*1000 + float(x[1])
    
    def height2num(self, x):
        if type(x) == str:
            x = x.strip('meters')
            return float(x)
    
    def wd2num(self, x):
        if type(x) == str:
            x = x.strip('cm')
            return float(x)

# change 'maker'
# NaN -- > Unknown, Maker names just keep M + 3 digits
def maker_change(x):
    if type(x) == str:
        x = x.split(',')
        s = ''
        for i in x:
            s += i[:4]
    else:
        s = 'UnKnown'
        
    return s

# change 'ingredient'
# use the number of ingredient
def ingredient_change(x):
    if type(x) == str:
        c = len(x.split(','))
    else:
        c = '0'
    return c


# data information check
# for numerical variable: show the numbers of missing data
# for categorical variable: get the info about missing value, type count
def data_info(df):
    print('----------Information of the data----------')
    print('The total rows of the data:', len(df))
    print('------------- Numerical data --------------')
    for i in ['cost', 'weight', 'height', 'width', 'depth']:
        print('{}: '.format(i), 'the number of missing values is', df[i].isnull().sum())
    print('------------- Categorical data ------------')
    for i in ['product_type', 'product_level', 'maker', 'ingredient']:
        print('{}: '.format(i), 'the number of missing values is', df[i].isnull().sum(),
              ' | ', 'the number of unique types is', df[i].nunique())

        
# check the collinearity of numerical variables
def plot_cor(df):
    '''
    check the relationships between numerical variables
    plot the correlation using heatmap
    plot the relationships using pairplot
    For variables 'price', 'cost', 'weight', 'height', 'width', 'depth'
    '''
    features = ['price', 'cost', 'weight', 'height', 'width', 'depth']
    
    sns.heatmap(df[features].corr(method='pearson'), annot=True)
    plt.title(
        'Correlations of numerical variables' 
        '\n Highly correlated: width-depth, width-height, height-depth'
        '\n Reasonably correlated: price-cost'
    )
    
    sns.pairplot(df[features])
    plt.suptitle('Relationships between numerical variables')
