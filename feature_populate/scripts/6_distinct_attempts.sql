-- Takes 300 seconds to execute
-- Created on July 1, 2013
-- @author: Franck for ALFA, MIT lab: franck.dernoncourt@gmail.com
-- edited by Colin Taylor on Feb 17, 2014
-- Feature 6: number of distinct problems attempted
-- (supported by http://francky.me/mit/moocdb/all/forum_posts_per_day_date_labels_cutoff120_with_and_without_cert.html)

set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;
set @start_date = 'START_DATE_PLACEHOLDER';

INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)

SELECT 6,
	user_dropout.user_id,
	FLOOR((UNIX_TIMESTAMP(submissions.submission_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) AS week,
	COUNT(DISTINCT(submissions.problem_id)),
  @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.submissions AS submissions
 ON submissions.user_id = user_dropout.user_id
WHERE user_dropout.dropout_week IS NOT NULL
	AND FLOOR((UNIX_TIMESTAMP(submissions.submission_timestamp)
			- UNIX_TIMESTAMP(@start_date)) / (3600 * 24 * 7)) < @num_weeks
  AND submissions.validity = 1
GROUP BY user_dropout.user_id, week
HAVING week < @num_weeks
AND week >= 0
;

