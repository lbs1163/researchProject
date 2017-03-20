function [tpr, fpr, tnr, fnr] = report_accuracy( confidences, label_vector )
% by James Hays

correct_classification = sign(confidences .* label_vector);
accuracy = 1 - sum(correct_classification <= 0)/length(correct_classification);
fprintf('  accuracy:   %.3f\n', accuracy);

true_positives = (confidences >= 0) & (label_vector >= 0);
n_tp = sum( true_positives );

false_positives = (confidences >= 0) & (label_vector < 0);
n_fp = sum( false_positives );

true_negatives = (confidences < 0) & (label_vector < 0);
n_tn = sum( true_negatives );

false_negatives = (confidences < 0) & (label_vector >= 0);
n_fn = sum( false_negatives );

tpr = n_tp/(n_tp+n_fn);
fpr = n_fp/(n_fp+n_tn);
tnr = n_tn/(n_tn+n_fp);
fnr = n_fn/(n_fn+n_tp);

fprintf('  true  positive rate: %.3f\n', tpr);
fprintf('  false positive rate: %.3f\n', fpr);
fprintf('  true  negative rate: %.3f\n', tnr);
fprintf('  false negative rate: %.3f\n', fnr);