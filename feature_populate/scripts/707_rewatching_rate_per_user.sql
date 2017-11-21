-- time the script takes to run on moocdb db: xxx seconds
-- date:  27/07/2017
-- author: John Mucong Ding
-- email:  johnding1996@hotmail.com
-- description of feature:

set @current_date = CAST('CURRENT_DATE_PLACEHOLDER' AS DATETIME);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;

DROP FUNCTION IF EXISTS get_if_rewatched;

CREATE FUNCTION get_if_rewatched (USER_ID VARCHAR(50), VIDEO_ID VARCHAR(2083))
    RETURNS FLOAT
    NOT DETERMINISTIC
BEGIN
  DECLARE if_rewatched FLOAT DEFAULT 0.0;
  DECLARE times INT DEFAULT 0;
  DECLARE loop_flag INT DEFAULT FALSE;
  DECLARE event_type VARCHAR(255) DEFAULT NULL;
  DECLARE cur_time FLOAT DEFAULT NULL;
  DECLARE prev_time FLOAT DEFAULT 0.0;
  DECLARE cur_time_stamp DATETIME;
  DECLARE prev_time_stamp DATETIME;
  DECLARE cur CURSOR FOR SELECT observed_event_type, video_current_time, observed_event_timestamp
    FROM `moocdb`.click_events AS click_events
    WHERE click_events.user_id = USER_ID AND click_events.video_id = VIDEO_ID
        -- Note this event types might be extended
        AND (observed_event_type = 'play_video' OR observed_event_type = 'pause_video')
    ORDER BY observed_event_timestamp ;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET loop_flag = TRUE;
  OPEN cur;
  cum_loop: LOOP
    FETCH cur INTO event_type, cur_time, cur_time_stamp;
    IF loop_flag THEN
      LEAVE cum_loop;
    END IF;
    IF TIMESTAMPDIFF(MINUTE, prev_time_stamp, cur_time_stamp) > 30 THEN
        SET times = times + 1;
    END IF;
    SET prev_time = cur_time;
    SET prev_time_stamp = cur_time_stamp;
  END LOOP;
  CLOSE cur;

  IF times > 0 THEN
    SET if_rewatched = 1.0;
  END IF;
  RETURN times;
END;


INSERT INTO `moocdb`.user_feature(feature_id, user_id, feature_value, date_of_extraction)
SELECT 707,
	outer_user_feature.user_id,
	(
	SELECT SUM(get_if_rewatched(outer_user_feature.user_id, video_feature.video_id))
	FROM `moocdb`.video_feature AS video_feature
    WHERE video_feature.feature_id = 704
	)
	/
	(
	SELECT feature_value
	FROM `moocdb`.user_feature AS user_feature
    WHERE user_feature.feature_id = 705 AND user_feature.user_id = outer_user_feature.user_id
	),
    @current_date
FROM `moocdb`.user_feature AS outer_user_feature
WHERE outer_user_feature.feature_id = 705
;