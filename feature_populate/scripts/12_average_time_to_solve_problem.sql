-- Takes 1000 seconds to execute
-- Created on July 2, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- Feature 12: average(max(attempt.timestamp) - min(attempt.timestamp) for each problem in a week)
-- 196966 rows
set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;
set @start_date = 'START_DATE_PLACEHOLDER';

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)

SELECT 12,
	user_dropout.user_id,
	FLOOR((UNIX_TIMESTAMP(submissions.submission_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) AS week,
	AVG(submissions.submission_attempt_number - submissions3.submission_attempt_number),
  @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.submissions AS submissions
 ON submissions.user_id = user_dropout.user_id
INNER JOIN `moocdb`.submissions AS submissions3
 ON submissions3.user_id = user_dropout.user_id
	AND submissions3.problem_id = submissions.problem_id
WHERE user_dropout.dropout_week IS NOT NULL
	AND submissions.submission_attempt_number = (
			SELECT MAX(submissions2.submission_attempt_number)
			FROM `moocdb`.submissions AS submissions2
			WHERE submissions2.problem_id = submissions.problem_id
				AND submissions2.user_id = submissions.user_id
		)
	AND submissions3.submission_attempt_number = (
			SELECT MIN(submissions2.submission_attempt_number)
			FROM `moocdb`.submissions AS submissions2
			WHERE submissions2.problem_id = submissions.problem_id
				AND submissions2.user_id = submissions.user_id
		)
	-- AND user_dropout.user_id <  1000
	AND FLOOR((UNIX_TIMESTAMP(submissions.submission_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) < @num_weeks
  AND submissions.validity = 1

GROUP BY user_dropout.user_id, week
HAVING week < @num_weeks
AND week >= 0
;
