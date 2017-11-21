-- Created on Feb 14, 2014
-- @author: Colin Taylor colin2328@gmail.com
-- Feature 208- number of attempts that were correct


set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;
set @start_date = 'START_DATE_PLACEHOLDER';


INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)


SELECT 208,
	user_dropout.user_id,
	FLOOR((UNIX_TIMESTAMP(submissions.submission_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) AS week,
	COUNT(*),
    @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.submissions AS submissions
	ON submissions.user_id = user_dropout.user_id
INNER JOIN `moocdb`.assessments
	ON assessments.submission_id = submissions.submission_id
WHERE user_dropout.dropout_week IS NOT NULL
AND assessments.assessment_grade = 1
AND submissions.validity = 1
GROUP BY user_dropout.user_id, week
HAVING week < @num_weeks
AND week >= 0;
