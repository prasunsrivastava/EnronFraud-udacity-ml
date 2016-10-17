# util.py
"""
this file contains functions for performing som eoperations on the 
data provided in the project folder. It defines functions for the 
following activities:

1. Create CSV file : A function is defined to create csv file from the
   data dict that is provided. This helps to read in the generated csv file
   easily with pandas and perform data exploration easily.
2. It defines functions for creating some new features related to email
   and finance data.

"""
import csv


def create_csv(data_dict, output_file):
    """ convert data dictionary to a csv file
        data_dict = a dict of dicts containing the
           data for the final project
        output_file = the path of the output csv
           file that is to be populated
    """
    dataset = []
    
    for name, feature_dict in data_dict.items():
        row = {}
        row['name'] = name
        for feature, feature_val in feature_dict.items():
            row[feature] = feature_val
        dataset.append(row)

    with open(output_file, "w") as output:
        writer = csv.DictWriter(output, dataset[0].keys())
        writer.writeheader()
        writer.writerows(dataset)

def create_email_features(data_dict, feature_list):
    """ mutates the feature_list and data_dict to contain new features.
        data_dict = data dictionary containing the data for the final
            project.
        feature_list = a list of feature names available in 
            data.
    """
    email_features = ['from_poi_to_this_person', 
                      'from_this_person_to_poi', 
                      'to_messages', 
                      'from_messages']
    
    for data in data_dict:
        person = data_dict[data]
        append = True
        
        for feature in email_features:
            if person[feature] == 'NaN':
                append = False
                break
        
        if append:
            total_messages = person['to_messages'] + person['from_messages']
            total_poi_msg = person['from_poi_to_this_person'] + person['from_this_person_to_poi']
            person['interaction_with_poi'] = total_poi_msg / float(total_messages)
        else:
            person['interaction_with_poi'] = 'NaN'
    
    feature_list += ['interaction_with_poi']

def create_binary_missing_features(data_dict, feature_list):
    """ mutates data_dict and feature_list to contain new 
        binary feature for each feature in the feature_list
        depicting the presence or absence of value in feature.
        data_dict = data dictionary containing the data for the
           final project
        feature_list = a list of features available in data
    """
    new_features = set()

    for data in data_dict:
        person = data_dict[data]

        for feature in feature_list:
            new_feature_name = 'missing_' + feature
            if person[feature] == 'NaN':
                person[new_feature_name] = 1
            else:
                person[new_feature_name] = 0
            new_features.add(new_feature_name)
    
    return new_features