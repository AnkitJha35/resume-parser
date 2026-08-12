class ResumeParserError(Exception):
    "Base error for resume parser failures."


class KafkaConsumerError(ResumeParserError):
    "Raised when the Kafka consumer fails to process messages."


class StorageClientError(ResumeParserError):
    "Raised when the storage client fails to download or read PDF data."
