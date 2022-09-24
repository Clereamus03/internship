'''
    Custom function tool for preprocessing .
    Author: AMIT PRATAP
    Company: Detect Technologies
'''
import pandas as pd
import matplotlib.pyplot as plt
import scipy as scp
# from plotnine import ggplot,aes,geom_point,stat_smooth,facet_wrap
import warnings
import streamlit as st
from sklearn.decomposition import PCA
import numpy as np
import ppscore as pps

class Preprocess:
    '''
    class for preprocessing the input data
    '''
    def __init__(self,df):
        '''intialization of dataframe'''
        self.df=df
        
    def dis_or_cont(self,feature):
        #better computationally expensive algo
        #(self.df[feature]).nunique()
        ''' 
        l=[]
        for i in self.df[feature]:
            if i not in l:
                l.append(i)
            else:
                continue
        '''
        if self.df[feature].nunique()<30:
            return 'dis'
        else:
            return 'cont'

    def eda_uni(self):
        '''
            f:feature name as str/'name of the feature'
            this function will be called in the eda_main function which will take the data from UI and operate on the 
            data , eda will come after the preprocessing is done i.e. after removal of missing values , imputation
             and before feature engineering and feature selection
        '''
        #y=pd.DataFrame(self.df[f],columns=[f])
        #line plot 
        #kind of plots : 'line' ,'box','scatter','pie'
        st.sidebar.header('select any one feature')
        feat=st.sidebar.selectbox('select the feature [Anyone]', self.df.columns, key="feature")
        #feat2=st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="second")
        kind=st.sidebar.selectbox("Select the kind of plot: ",['Select One', 'line' ,'pie','box','hist','scatter'])
        if kind =='line':
            #fig=plt.figure(figsize=(15,10))
            plt.plot(self.df[feat])
            #plt.plot(self.df[feat2])
            plt.xlabel([feat])
            plt.ylabel("Values")
            plt.legend()
            st.pyplot()
        elif kind=='hist':
            plt.figure(figsize=(15,10))
            plt.hist(self.df[feat])
            plt.xlabel([feat])
            plt.ylabel("Values")
            plt.legend()
            st.pyplot()

            
        elif kind=='box':
            plt.figure(figsize=(15,10))
            #ax=fig.add_subplot(111)
            plt.boxplot([self.df[feat]],notch=True)
            #ax.set_yticklabels([self.df[feat1],self.df[feat2]])
            #ax.get_xaxis().tick_bottom()
            plt.xlabel([feat])
            plt.ylabel("Values")
            plt.legend()
            #plt.show()
            st.pyplot()
        
        elif kind=='scatter': 
            #fig=plt.figure(figsize=(15,10))
            #read docs on plt.plot.scatter 
            plt.scatter(self.df.index,self.df[feat])
            plt.xlabel([feat])
            plt.ylabel("Values")
            plt.legend()
            #plt.show()
            st.pyplot()
        
            #pie chart code 
        elif kind=='pie':
            #pie chart analysis ...syntax error rectification
            plt.figure(figsize=(15,10))
            label=[]
            for i in self.df[feat]:
                if i not in label:
                    label.append(i)
                else:
                    continue
            
            plt.pie(self.df[feat].value_counts(),labels=label)
            #plt.pie(self.df[feat],label)
            #plt.legend()
            #plt.show()
            #plt.xlabel(feat)
            #plt.ylabel("values")
            st.pyplot()
    def eda_bi(self,feat1,feat2):
        '''for the purpose of inferencing whether the data follows a certain observable pattern .
        One set of results will go to the visualization part and it will be savedd later in datawarehouse to generate results after the 
        preprocessing of the data 
        Format of feat1 and feat2 would be a string format .
        '''
        #self.df.drop(dep_feat,axis=1,inplace=True)
        """
        for i in range(len(self.df.columns)):
            plt.figure(figsize=(15,10))
            self.df.iloc[:,i].plot(kind = 'scatter', label = self.df.columns[i])
            plt.legend()
            plt.show()
        
        st.sidebar.header('select any one feature')
        feat=st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="feature")
        #feat2=st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="second")
        kind=st.sidebar.selectbox("Select the kind of plot: ",['Select One', 'line' ,'pie','box','hist','scatter'])
        """
        #t1=self.dis_or_cont(feat1)
        #t2=self.dis_or_cont(feat2)
        kind=st.sidebar.selectbox("Select the kind of plot: ",['Select One', 'line' ,'area','bar','scatter','box'])
        if kind=='line':
            plt.figure(figsize=(15,10))
            plt.plot(self.df[feat1],label=feat1)
            plt.plot(self.df[feat2],label=feat2)
            plt.xlabel([feat1,feat2])
            plt.ylabel("Values")
            plt.legend()
            st.pyplot()
        elif kind=='bar':
            plt.figure(figsize=(15,10))
            plt.bar(self.df.index,height=self.df[feat1],width=.2)
            plt.bar(self.df.index,height=self.df[feat2],bottom=self.df[feat1],width=0.2)
            plt.xlabel([feat1,feat2])
            plt.ylabel("Values")
            plt.legend()
            #plt.show()
            st.pyplot()
        elif kind=='area':
            plt.figure(figsize=(15,10))
            plt.stackplot(self.df.index,self.df[feat1], self.df[feat2], labels=[feat1,feat2])
            plt.legend(loc='upper left')    
            plt.legend()
            st.pyplot()
        elif kind=='box':
            plt.figure(figsize=(15,10))
            plt.boxplot([self.df[feat1],self.df[feat2]],notch=True)
            #plt.boxplot([self.df[feat1]],notch=True)
            #plt.boxplot([self.df[feat2]],notch=True)
            plt.xlabel([feat1,feat2])
            plt.ylabel("Values")
            plt.legend()
            #plt.show()
            st.pyplot()
        elif kind=='scatter':
            plt.figure(figsize=(15,10))
            plt.scatter(self.df[feat1],self.df[feat2])
            #plt.scatter(self.df.index,self.df[feat1],label=feat1)
            #plt.scatter(self.df.index,self.df[feat2],label=feat2)
            plt.xlabel(feat1)
            plt.ylabel(feat2)
            plt.legend()
            st.pyplot()

    '''
    def eda_multi(self,feat1,feat2,color_diff_factor,num_of_plots):
        """
            feat1 and feat1 coud be continuous, and color_diff_factor and num_of_plots are the variables which 
            are supposed to different colors and define number of plots - both are categorical ideally
            First two variables will define the x labels and y labels - the other two variables will contribute to
            the color differentiation and number of plots respectively .
        """
        a=(f"factor({color_diff_factor})")
        b=(f"~{num_of_plots}")
        
        return (ggplot(self.df, aes(feat1, feat2, color=a))
        + geom_point()
        + stat_smooth(method='auto')
        + facet_wrap(b))
        
        p = ggplot(self.df, aes(feat1, feat2, color=a))+ geom_point()+ stat_smooth(method='auto')+ facet_wrap(b)
        return st.pyplot(ggplot.draw(p))
    '''
    def eda_multi(self,feat1,feat2,feat3):
        #multivariate , kind ='stacked line','3D plot' , ggplot
        kind=st.sidebar.selectbox("Select the kind of plot: ",['Select One', '3D','line' ,'box'])
        if kind=='line':
            plt.figure(figsize=(15,10))
            plt.plot(self.df[feat1],label=feat1)
            plt.plot(self.df[feat2],label=feat2)
            plt.plot(self.df[feat3],label=feat3)
            plt.xlabel([feat1,feat2,feat3])
            plt.ylabel("Values")
            plt.legend()
            st.pyplot()
        elif kind=='box':
            plt.figure(figsize=(15,10))
            plt.boxplot([self.df[feat1],self.df[feat2],self.df[feat3]],notch=True)
            #plt.boxplot([self.df[feat1]],notch=True)
            #plt.boxplot([self.df[feat2]],notch=True)
            plt.xlabel([feat1,feat2,feat3])
            plt.ylabel("Values")
            plt.legend()
            #plt.show()
            st.pyplot()
        
        elif kind=='3D':
            ax = plt.axes(projection='3d')
            # Data for a three-dimensional line
            zline = self.df[feat1]
            xline = self.df[feat2]
            yline = self.df[feat3]
            ax.plot3D(xline, yline, zline, 'gray')
            # Data for three-dimensional scattered points
            #zdata = 15 * np.random.random(100)
            #xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
            #ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
            ax.scatter3D(xline, yline, zline, c=zline, cmap='Greens')
            ax.set_xlabel(feat2)
            ax.set_ylabel(feat3)
            ax.set_zlabel(feat1)
            st.pyplot()



    def feature_importance_calculation(self,dep_var):
        """
            PPS :Predecitive Power Scores
            calculates non linear relationships between the variables given 
        """
        import operator
        scores={}
        for i in self.df.columns:
            if i!=dep_var:
                scores[(i)]=pps.score(self.df,i,dep_var)['ppscore']
        sorted_scores=sorted(scores.items(), key=operator.itemgetter(1),reverse=True)
        return sorted_scores[0][0],sorted_scores[1][0],sorted_scores[2][0],sorted_scores[3][0]
        
    def eda_default(self):
        """
            plots of features which are important 
            line, bar , histogram- based on whether the data is continuous or categorical

        """

        pass
    def eda_main(self, feat1,feat2,type_of_plot):
        '''
            main function which will be called in the main.py or app.py
            type_of_plot=univariate , bivariate , multivariate 
        '''

        pass

    def count_nans(self,dep_feat):
        '''removal of sparse info or data columns/dropping of feature columns.
            Problem with dependent features'''
        count=[]
        d=self.df[dep_feat]
        self.df.drop(dep_feat,axis=1,inplace=True)
        for i in self.df.columns:
            count.append(self.df[i].isnull().sum()/len(self.df[i]))
        for i,j in enumerate(count):
            if j >=0.8:
                self.df.drop(self.df.columns[i],axis=1,inplace=True)
        self.df[d.name]=d
        return self.df

    def imputation(self):
        ''' k means , mean, mode, median, target_mean_encoding,interpolation-linear or polynomial etc.'''

        pass
    def feature_selection(self):
        '''PCA , Correlation'''


        pass
    def feature_engineering(self):
        '''addition of new features and transformations'''
        pass
    def encoder(self):
        #col=list(self.df.columns)
        lst = []
        transform_dict = {}


        # Appending the names of Categorical columns to lst
        for i in self.df.columns:
            if self.df[i].dtype == "O" :
                lst.append(self.df[i].name)

        # This will add Categorical columns with their nested dictionary of categories with their labels
        for col in lst:
            cats = pd.Categorical(self.df[col]).categories
            d = {}
            for i, cat in enumerate(cats):
                d[cat] = i
            transform_dict[col] = d
            
        self.df = self.df.replace(transform_dict)
        
        # with open("files/encoder.pkl", "wb") as f:
        #     pickle.dump(transform_dict, f)
        
        return self.df