-- MySQL Script generated by MySQL Workbench
-- Sat Apr 27 05:11:48 2024
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema fyp
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema fyp
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `fyp` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `fyp` ;


-- -----------------------------------------------------
-- Table `fyp`.`trending_topics` REPLACE PREVIOUS onee
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`trending_topics` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `TopicKeyword` VARCHAR(128) NOT NULL,
  `Topic` VARCHAR(512) NOT NULL,
  PRIMARY KEY (`id`))
select * from trending_topics;

-- -----------------------------------------------------
-- Table `fyp`.`admins`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`admins` (
  `admin_id` INT NOT NULL AUTO_INCREMENT,
  `admin_name` VARCHAR(50) NOT NULL,
  `admin_password` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`admin_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`universities`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`universities` (
  `university_id` INT NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`university_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`domains`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`domains` (
  `domain_id` INT NOT NULL AUTO_INCREMENT,
  `university_id` INT NOT NULL,
  PRIMARY KEY (`domain_id`),
  INDEX `university_id` (`university_id` ASC) VISIBLE,
  CONSTRAINT `domains_ibfk_1`
    FOREIGN KEY (`university_id`)
    REFERENCES `fyp`.`universities` (`university_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`filters`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`filters` (
  `filter_id` INT NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`filter_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`mentors`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`mentors` (
  `mentor_id` INT NOT NULL AUTO_INCREMENT,
  `Name` VARCHAR(255) NULL DEFAULT NULL,
  `UniversityName` VARCHAR(255) NULL DEFAULT NULL,
  `Email` VARCHAR(255) NULL DEFAULT NULL,
  `ResearchInterests` VARCHAR(255) NULL DEFAULT NULL,
  `Designation` VARCHAR(255) NULL DEFAULT NULL,
  `Country` VARCHAR(255) NULL DEFAULT NULL,
  PRIMARY KEY (`mentor_id`))
ENGINE = InnoDB
AUTO_INCREMENT = 7662
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`users`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(25) NOT NULL,
  `password` VARCHAR(25) NOT NULL,
  `email` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB
AUTO_INCREMENT = 2
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `fyp`.`messages`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`messages` (
  `msgid` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `email_from` VARCHAR(50) NOT NULL,
  `message` TEXT NOT NULL,
  `sent_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`msgid`),
  INDEX `user_id` (`user_id` ASC) VISIBLE,
  CONSTRAINT `messages_ibfk_1`
    FOREIGN KEY (`user_id`)
    REFERENCES `fyp`.`users` (`id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `fyp`.`newsletter_subscribers`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`newsletter_subscribers` (
  `subscriber_id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(50) NOT NULL,
  `email` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`subscriber_id`),
  UNIQUE INDEX `email` (`email` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`reviews`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`reviews` (
  `userid` INT NOT NULL,
  `reviewid` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(25) NOT NULL,
  `review` VARCHAR(250) NOT NULL,
  PRIMARY KEY (`reviewid`),
  INDEX `userid` (`userid` ASC) VISIBLE,
  CONSTRAINT `reviews_ibfk_1`
    FOREIGN KEY (`userid`)
    REFERENCES `fyp`.`users` (`id`))
ENGINE = InnoDB
AUTO_INCREMENT = 2
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `fyp`.`searches`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`searches` (
  `search_id` INT NOT NULL AUTO_INCREMENT,
  `mentor_id` INT NOT NULL,
  `domain_id` INT NOT NULL,
  `filter_id` INT NOT NULL,
  PRIMARY KEY (`search_id`),
  INDEX `mentor_id` (`mentor_id` ASC) VISIBLE,
  INDEX `domain_id` (`domain_id` ASC) VISIBLE,
  INDEX `filter_id` (`filter_id` ASC) VISIBLE,
  CONSTRAINT `searches_ibfk_1`
    FOREIGN KEY (`mentor_id`)
    REFERENCES `fyp`.`mentors` (`mentor_id`),
  CONSTRAINT `searches_ibfk_2`
    FOREIGN KEY (`domain_id`)
    REFERENCES `fyp`.`domains` (`domain_id`),
  CONSTRAINT `searches_ibfk_3`
    FOREIGN KEY (`filter_id`)
    REFERENCES `fyp`.`filters` (`filter_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`students`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`students` (
  `student_id` INT NOT NULL AUTO_INCREMENT,
  `student_name` VARCHAR(50) NOT NULL,
  `student_email` VARCHAR(50) NOT NULL,
  `student_password` VARCHAR(50) NOT NULL,
  `university_id` INT NOT NULL,
  `search_id` INT NOT NULL,
  PRIMARY KEY (`student_id`),
  INDEX `university_id` (`university_id` ASC) VISIBLE,
  INDEX `search_id` (`search_id` ASC) VISIBLE,
  CONSTRAINT `students_ibfk_1`
    FOREIGN KEY (`university_id`)
    REFERENCES `fyp`.`universities` (`university_id`),
  CONSTRAINT `students_ibfk_2`
    FOREIGN KEY (`search_id`)
    REFERENCES `fyp`.`searches` (`search_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `fyp`.`subscribers`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`subscribers` (
  `emailid` INT NOT NULL AUTO_INCREMENT,
  `email` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`emailid`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb3;


-- -----------------------------------------------------
-- Table `fyp`.`trending_topics` DO NOT USE THIS ONE!
-- -----------------------------------------------------
/* CREATE TABLE IF NOT EXISTS `fyp`.`trending_topics` (
  `IOT` VARCHAR(255) NULL DEFAULT NULL,
  `Machine learning` VARCHAR(255) NULL DEFAULT NULL,
  `digital forensics` VARCHAR(255) NULL DEFAULT NULL,
  `Blockchain` VARCHAR(255) NULL DEFAULT NULL,
  `Vehicular Ad Hoc Networks` VARCHAR(255) NULL DEFAULT NULL,
  `Wireless Sensor Networks` VARCHAR(255) NULL DEFAULT NULL,
  `Cloud Computing` VARCHAR(255) NULL DEFAULT NULL,
  `Fog Computing` VARCHAR(255) NULL DEFAULT NULL,
  `Edge Computing` VARCHAR(255) NULL DEFAULT NULL,
  `Cloud Security` VARCHAR(255) NULL DEFAULT NULL,
  `Mobile Cloud Computing (MCC)` VARCHAR(255) NULL DEFAULT NULL,
  `Data Mining` VARCHAR(255) NULL DEFAULT NULL,
  `Big Data` VARCHAR(255) NULL DEFAULT NULL,
  `Web Technology` VARCHAR(255) NULL DEFAULT NULL,
  `Mobile Ad Hoc Networks (MANET)` VARCHAR(255) NULL DEFAULT NULL)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci; */


-- -----------------------------------------------------
-- Table `fyp`.`user_accounts`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fyp`.`user_accounts` (
  `user_id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(50) NOT NULL,
  `email` VARCHAR(50) NOT NULL,
  `password` VARCHAR(50) NOT NULL,
  `user_type` ENUM('student', 'admin') NULL DEFAULT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE INDEX `username` (`username` ASC) VISIBLE,
  UNIQUE INDEX `email` (`email` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;