import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd


class Results:
    def __init__(self, filename_seg, filename_patient):
        '''
        :param filename_seg:  Filename  (.csv) where to save results at the segment levels
        :param filename_patient: Filename  (.csv) where to save results at the patient levels
        '''
        self.results_patients = np.zeros(3)
        self.results_segments = np.zeros(3)
        self.filename_seg = filename_seg
        self.filename_patient = filename_patient

    def add_result(self, res, accuracy, segments=True):
        '''

        :param res: result of classification report (sklearn )
        :param accuracy:
        :param segments: 1 to add results at the segment level
        :return:
        '''
        if segments:
            specificity = res['0.0']['recall']
            sensitivy = res['1.0']['recall']
        else:
            specificity = res['0']['recall']
            sensitivy = res['1']['recall']
        all = np.array([specificity, sensitivy, accuracy])

        if segments:
            self.results_segments = np.vstack((self.results_segments, all))
        else:
            self.results_patients = np.vstack((self.results_patients, all))

    def validate_patient(self, model, x_val, y_val, count):
        '''

        :param model: trained model after 1 fold of cross validation
        :param x_val: x_Val for 1 forld of cross validation
        :param y_val: y_Val for 1 forld of cross validation
        :param count: vector containing the number of segments per patient
        :return:  save the results of the fold

        '''
        ## per segments
        pred_seg = model.predict(np.split(x_val, x_val.shape[2], axis=2))
        res = classification_report(np.rint(y_val), np.rint(pred_seg), output_dict=True)
        acc = accuracy_score(np.rint(y_val), np.rint(pred_seg))
        self.add_result(res, acc, True)

        eval = []
        y = []
        pred = []
        for m in range(1, len(count)):
            i = count[m]
            j = count[m - 1]
            score = model.evaluate(np.split(x_val[j:i, :, :], x_val.shape[2], axis=2), y_val[j:i])
            eval.append(score)
            y.append(np.int(np.mean(y_val[j:i])))
            p = np.rint(model.predict(np.split(x_val[j:i, :, :], x_val.shape[2], axis=2)))
            pred.append(np.mean(p))

        res = classification_report(y, np.rint(pred), output_dict=True)
        print(classification_report(y, np.rint(pred)))

        acc = accuracy_score(np.rint(y), np.rint(pred))
        self.add_result(res, acc, False)

        # np.savetxt(self.filename_patient, self.results_patients, delimiter=",")
        # np.savetxt(self.filename_seg, self.results_segments, delimiter=",")
        res_segments_dict = {'Specificity': self.results_segments[1:, 0], 'Sensitivity': self.results_segments[1:, 1],
                             'Accuracy': self.results_segments[1:, 2]}
        df = pd.DataFrame.from_dict(res_segments_dict)
        df.to_csv(self.filename_seg)
        res_patients_dict = {'Specificity': self.results_patients[1:, 0], 'Sensitivity': self.results_patients[1:, 1],
                             'Accuracy': self.results_patients[1:, 2]}
        df = pd.DataFrame.from_dict(res_patients_dict)
        df.to_csv(self.filename_patient)
