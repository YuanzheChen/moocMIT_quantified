-- time the script takes to run on moocdb db: xxx seconds
-- date:  25/07/2017
-- author: John Mucong Ding
-- email:  johnding1996@hotmail.com
-- description of feature:

set @current_date = CAST('CURRENT_DATE_PLACEHOLDER' AS DATETIME);
set @num_weeks = NUM_WEEKS_PLACEHOLDER;

DROP FUNCTION IF EXISTS get_scroll_back_times;

CREATE FUNCTION get_scroll_back_times (USER_ID VARCHAR(50), VIDEO_ID VARCHAR(255))
    RETURNS FLOAT
    NOT DETERMINISTIC
BEGIN
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
    -- Remember the order of two variables in TIMESTAMPDIFF func is important
    IF TIMESTAMPDIFF(MINUTE, prev_time_stamp, cur_time_stamp) <= 30 THEN
        IF prev_time > cur_time THEN
            SET times = times + 1;
        END IF;
    END IF;
    SET prev_time = cur_time;
    SET prev_time_stamp = cur_time_stamp;
  END LOOP;
  CLOSE cur;

  RETURN times;
END;

INSERT INTO `moocdb`.user_video_feature(feature_id, user_id, video_id, date_of_extraction)
SELECT 703,
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
SET feature_value = get_scroll_back_times(user_id, video_id)
WHERE feature_id = 703
;