# Starbucks Offer Recommendation System

## Introduction

This project was done as a requirement for the Udacity Data Scientist
Nanodegree. The goal of the project is to use the transactional,
customer and offer data to provide offer recommendations to Starbucks
customers. The data was obtained by a simulation of the Starbucks
mobile app in which customers receive and view offers, and pay their
drinks in the stores.

## Configuration

The project was implemented using Jupyter Notebooks, Python and
libraries used for data analysis such as Pandas and Numpy. 

## Methodology

In this project, we use the techniques used in a regular data science project.

### 1. Business Understanding

To guide the project, the following questions were considered:

- What are the main factors that contribute to customers making purchases?
- Are offers a way to increase customer engagement?
- What kind of offers are the most popular?
- What populations are more interested in offers?
- What offers should we recommend to different customers?

### 2. Prepare Data

In the [helper.py](https://github.com/MohanCR97/Udacity-Data_Scientist_Nanodegree/blob/master/Starbucks_CapstoneProject/helper.py)
file, the `clean_portfolio`, `clean_profile` and `clean_transcript` functions are
provided. They implement the following functionality:

**Portfolio Dataframe Tasks**
* Split the channels into several columns
* Split offer_type into several columns
* change id column name to offer_id

**Profile Dataframe Tasks**
* Fix the date.
* Split gender column into dummy columns
* Change the column name id to customer_id. 

**Transcript Dataframe Tasks**

* Split value in several columns for offers and transactions
* Split event column into several columns
* Change column name person to customer_id


### 3. Exploratory Analysis

In this stage, we analyzed the population based on their demographics
and their spending behavior. We also took into account the
interactions between the customers and the offers provided.


### 4. Recommendation Engine

We used a knowledge based recommendation engine in this project. We
provided one that selects the most popular offers without considering
demographics, first. This system is a good start for customers that do
not provide any demographic data in the app.

For the rest of costumers, we introduced filters that help the system
make recommendations based on the demographic data provided by the
customers.

We evaluated the recommendation systems by using visualizations from
the data.

### 5. Blog Website

A blog has been published in this [site](https://aproano2.github.io/starbucks/)
