-- time the script takes to run on moocdb db: xxx seconds
-- date:  26/07/2017
-- author: John Mucong Ding
-- email:  johnding1996@hotmail.com
-- description of feature:

set @current_date = CAST('CURRENT_DATE_PLACEHOLDER' AS DATETIME);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;

INSERT INTO `moocdb`.user_feature(feature_id, user_id, feature_value, date_of_extraction)
SELECT 705,
	user_dropout.user_id,
	COUNT(DISTINCT(click_events.video_id)),
    @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.click_events AS click_events
    ON click_events.user_id = user_dropout.user_id
WHERE user_dropout.dropout_week IS NOT NULL
GROUP BY user_dropout.user_id
;