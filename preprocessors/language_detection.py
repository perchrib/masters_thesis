import langdetect as ld

ENGLISH = 'en'
SPANISH = 'es'
GERMAN = 'de'
DUTCH = 'nl'

def detect_language(text, language=SPANISH, ret_conf=False):
    """
    Uses langdetect to detect a language in text. Spanish by default
    :param text: text to analyze
    :param language: language to detect
    :param ret_conf: return dict of language confidences or not
    :return:
    """

    try:
        detections = ld.detect_langs(text)
        confidences = {lang_obj.lang: lang_obj.prob for lang_obj in detections}

        if language in confidences and confidences[language] > 0.9:
            # print(text)
            if not ret_conf:
                return True

        elif ret_conf:
            return confidences
        else:
            return False

    except Exception:
        # print("No language detected - Probably URL")
        return False

if __name__ == '__main__':
    # Num trials
    for i in range(10):
        print(detect_language("Ik ben voor", ret_conf=True))