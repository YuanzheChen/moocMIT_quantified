-- time the script takes to run on moocdb db: xxx seconds
-- date:  01/08/2017
-- author: John Mucong Ding
-- email:  johnding1996@hotmail.com
-- description of feature:

set @current_date = CAST('CURRENT_DATE_PLACEHOLDER' AS DATETIME);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;

INSERT INTO `moocdb`.user_feature(feature_id, user_id, feature_value, date_of_extraction)
SELECT 714,
	user_video_feature.user_id,
    SUM(feature_value),
    @current_date
FROM `moocdb`.user_video_feature AS user_video_feature
WHERE user_video_feature.feature_id=703
GROUP BY user_video_feature.user_id
;