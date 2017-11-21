-- Created on Feb 3rd, 2014
-- @author:  Colin Taylor
-- Feature 207: Lab Grade over time: lab grade - avg (lab grade from previos weeks)
-- Meant to be run in order to run after populate_feature_206_pset_grade.sql

DROP PROCEDURE IF EXISTS `moocdb`.Populate_207;
CREATE PROCEDURE `moocdb`.Populate_207()
    BEGIN
        DECLARE x  INT;
        SET x = 1;
        set @current_date = cast('CURRENT_DATE_PLACEHOLDER' as datetime);
        WHILE x  <= 13 DO
            INSERT INTO `moocdb`.user_long_feature(feature_id, user_id, feature_week, feature_value,date_of_extraction)
            SELECT 207, d1.user_id, x AS week, d1.feature_value -
                (SELECT AVG(feature_value)
                FROM `moocdb`.user_long_feature AS d2 WHERE feature_id = 206 AND feature_week < x AND d1.user_id = d2.user_id) AND date_of_extraction >= @current_date,
                @current_date
            FROM `moocdb`.user_long_feature AS d1 WHERE feature_id = 206 AND feature_week = x AND date_of_extraction >= @current_date;
            SET  x = x + 1;
        END WHILE;
    END;

CALL `moocdb`.Populate_207();
