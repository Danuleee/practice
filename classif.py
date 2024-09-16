import re

class DocumentClassifier:
    def classify_document(self, ocr_text):
        self.ocr_text = ocr_text.lower()  # Приводим текст OCR к нижнему регистру
        
        if self.is_passport():
            return "Passport"
        elif self.is_contract():
            return "Contract"
        elif self.is_birth_certificate():
            return "Birth Certificate"
        elif self.is_id_card():
            return "ID Card"
        elif self.is_agreement():
            return "Agreement"
        else:
            return "Unknown Document"

    def is_passport(self):
        # Ищем паттерны, характерные для паспорта
        passport_patterns = [
            #r'\b[a-z]{1,2}\d{6,9}\b',  # Серийный номер паспорта
            r'\bpassport\b'        # Дата рождения
        ]
        return self.contains_patterns(passport_patterns)

    def is_contract(self):
        # Ищем паттерны, характерные для контракта
        contract_patterns = [
            r'\bcontract\b',            # Упоминание контракта
        ]
        return self.contains_patterns(contract_patterns)

    def is_birth_certificate(self):
        # Ищем паттерны, характерные для свидетельства о рождении
        birth_certificate_patterns = [
            r'\bbirth\b',   # Свидетельство о рождении
        ]
        return self.contains_patterns(birth_certificate_patterns)

    def is_id_card(self):
        # Ищем паттерны, характерные для удостоверения личности (ID)
        id_patterns = [
            r'\bid card\b',             # Удостоверение личности
            r'\bidentification number\b',# Идентификационный номер
            r'\bnationality\b',         # Гражданство
            #r'\bdate of birth\b'        # Дата рождения
        ]
        return self.contains_patterns(id_patterns)

    def is_agreement(self):
        # Ищем паттерны, характерные для соглашений
        agreement_patterns = [
            r'\bagreement\b',           # Упоминание соглашения
            r'\bterms\b',               # Условия соглашения
            r'\bright\b',               # Права
            r'\bobligations\b'          # Обязанности
        ]
        return self.contains_patterns(agreement_patterns)

    def contains_patterns(self, patterns):
        # Поиск совпадений с регулярными выражениями
        return any(re.search(pattern, self.ocr_text) for pattern in patterns)

# Пример использования:
#classifier = DocumentClassifier()
#ocr_text = "Your contract is ready to be signed by both parties."
#print(classifier.classify_document(ocr_text))