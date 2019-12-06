# Identify Customer Segment

## Description
Analyze real-life data concerning a company that performs mail-order sales in Germany. The data is provided by Bertelsmann partners AZ Direct and Arvato Financial Solution. Demographics data from the mail-order firm's customers (191,652 individuals) and a subset of the German population (891,211 individuals) are provided for this analysis. In total, there are 85 demographic features available. 

The goal of the project is to apply unsupervised learning techniques to identify segments of the German population that are popular with the mail-order firm.

The resulting cluster analysis can be used for tasks such as identifying which facets of the population are likely to purchase the firm's products for a mailout campaign. 

## Data
1. Udacity_AZDIAS_Subset.csv: Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
2. Udacity_CUSTOMERS_Subset.csv: Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
3. Data_Dictionary.md: Information file about the features in the provided datasets.
4. AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographic data.

Due to agreements with Bertelsmann, the datasets cannot be shared or used for used for tasks other than this project. Files have to be removed from the computer within 2 weeks after completion of the project.

## Methodology
1. Start with analyzing demographics data of general population. Deal with missing values in the dataset.
2. Drop variables with extremely high frequencies of missing values.
3. Drop rows/individuals with high amount of missing values. Noted that these individuals have relatively different distribution of data values on columns that are not missing data as compared to the rest of the population.
4. Re-encode categorical features to dummy or one-hot encoded features.
5. Identify mixed-type features and engineer new features from them. A mixed-type feature generally tracks information on two or more dimensions. For example, "CAMEO_INTL_2015" variable combines information on two axes: wealth and life stage. In this case, we will generate two new features: one tracking wealth and the other tracking life stage.
6. Perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Use StandardScaler to ccale each feature to mean 0 and standard deviation 1.
7. Perform dimensionality reduction using PCA. Identify which demographics attribute are the most positively or negatively correlated with each principal component. This will allow us to interpret the principal components.
8. Apply clustering to general population with KMeans.
9. Repeat steps 1-8 on the demographics data of the firm's customer, using the same StandardScaler, PCA and KMeans objects fitted to the general population.
10. Analyze the cluster distributions of the general population and the firm's customers. Identify clusters that are overrepresented or underrepresented in the customer dataset compared to the general population.
11. Identify what kinds of people are typified by these overrepresented or underrepresented clusters. For each cluster, identify which principal components have the highest means and isolate them. Using the interpretations of the principal components from step 7, analyze what demographics attributes the top principal components are capturing. Use the information to deduce the demographic attributes of the overrepresented and underrepresented clusters.

Results are published in ipynb files

## How To Run It
Run identify_customer_segments.ipynb

## Installations
Anaconda, Seaborn
