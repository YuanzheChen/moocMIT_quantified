-- time the script takes to run on moocdb db: xxx seconds
-- date:  10/07/2017
-- author: John Mucong Ding
-- email:  johnding1996@hotmail.com
-- description of feature:

set @current_date = CAST('CURRENT_DATE_PLACEHOLDER' AS DATETIME);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;

DROP FUNCTION IF EXISTS get_session_engagement_time;

CREATE FUNCTION get_session_engagement_time (USER_ID VARCHAR(50), VIDEO_ID VARCHAR(255))
    RETURNS FLOAT
    NOT DETERMINISTIC
BEGIN
  DECLARE cum_time FLOAT DEFAULT 0.0;
  DECLARE loop_flag INT DEFAULT FALSE;
  DECLARE event_type VARCHAR(255) DEFAULT NULL;
  DECLARE cur_time FLOAT DEFAULT NULL;
  DECLARE prev_time FLOAT DEFAULT 0.0;
  DECLARE start_time FLOAT DEFAULT NULL;
  DECLARE end_time FLOAT DEFAULT NULL;
  DECLARE count_flag INT DEFAULT FALSE;
  DECLARE cur_time_stamp DATETIME;
  DECLARE prev_time_stamp DATETIME;
  DECLARE cur CURSOR FOR SELECT observed_event_type, video_current_time, observed_event_timestamp
    FROM `moocdb`.click_events AS click_events
    WHERE click_events.user_id = USER_ID AND click_events.video_id = VIDEO_ID
        AND (observed_event_type = 'play_video' OR observed_event_type = 'pause_video')
    ORDER BY observed_event_timestamp ;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET loop_flag = TRUE;
  SET cum_time = 0.0;
  OPEN cur;
  cum_loop: LOOP
    FETCH cur INTO event_type, cur_time, cur_time_stamp;
    IF loop_flag THEN
      LEAVE cum_loop;
    END IF;
    IF count_flag THEN
        IF prev_time > cur_time THEN
            SET end_time = prev_time;
            SET cum_time = cum_time + end_time - start_time;
            SET count_flag = FALSE;
        ELSEIF event_type = 'pause_video' THEN
            SET end_time = cur_time;
            SET cum_time = cum_time + end_time - start_time;
            SET count_flag = FALSE;
        -- Now only count the engagement time of the last watching session
        -- This is limited by the fact that a stored function can only return a number, not a table
        -- TODO: extend to count on all sessions
        ELSEIF TIMESTAMPDIFF(MINUTE, prev_time_stamp, cur_time_stamp) > 30 THEN
            SET cum_time = 0.0;
            SET count_flag = FALSE;
        END IF;
    ELSE
        IF event_type = 'play_video' THEN
            SET start_time = cur_time;
            SET count_flag = TRUE;
        END IF;
    END IF;
    SET prev_time = cur_time;
    SET prev_time_stamp = cur_time_stamp;
  END LOOP;
  CLOSE cur;

  RETURN cum_time;
END;

INSERT INTO `moocdb`.user_video_feature(feature_id, user_id, video_id, date_of_extraction)
SELECT 702,
	user_dropout.user_id,
	click_events.video_id,
    @current_date
FROM `moocdb`.user_dropout AS user_dropout
INNER JOIN `moocdb`.click_events AS click_events
    ON click_events.user_id = user_dropout.user_id
WHERE user_dropout.dropout_week IS NOT NULL
GROUP BY user_dropout.user_id, click_events.video_id
;

UPDATE `moocdb`.user_video_feature
SET feature_value = get_session_engagement_time(user_id, video_id)
WHERE feature_id = 702
;