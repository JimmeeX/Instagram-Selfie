"""
Special Utility Functions for MARK5828 Week 2 Tutorial Content. The functions saved here are unnecessary for the students to learn, but they can read if they are curious.
Author: James Lin
Special Thanks: Daniel-Han Chan
"""

import itertools
from scipy.linalg.blas import ssyrk
from scipy.linalg.lapack import spotrf, sposv
import numpy as np
import pandas as pd

import urllib.parse
import requests

def convert_to_seconds(time_str):
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + round(float(s))

def get_source_language_v2(data):
    return data["videos"][0]["insights"]["sourceLanguage"]

def get_video_duration_v2(data):
    return data["summarizedInsights"]["duration"]["seconds"]

def get_faces_v2(data):
    face_names = []
    face_duration = []

    for i, face in enumerate(data["summarizedInsights"]["faces"]):
        face_names.append(face["name"])
        face_duration.append(round(face["seenDuration"]))

    faces = dict(zip(face_names, face_duration))
    sorted_faces = dict(sorted(faces.items(), key=lambda x: x[1], reverse=True))

    return sorted_faces

def get_topics_v2(data):
    """Extracts Topics from the Analysed Video using Video Indexer version 2.

    Arguments:
        data {[json]} -- [Video Indexer Output]

    Returns:
        [Dictionary - (Topic Name: Topic Duration)] -- [Ranked from highest to lowest score]
    """
    topic_names = []
    topic_durations = []
    for topic in data["summarizedInsights"]["topics"]:
        topic_names.append(topic["name"])
        duration = 0
        for item in topic["appearances"]:
            duration += (item["endSeconds"] - item["startSeconds"])
        topic_durations.append(round(duration))
    topics = dict(zip(topic_names, topic_durations))
    sorted_topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

    return sorted_topics

def get_keywords_v2(data):
    """Extracts Keywords from the Analysed Video using Video Indexer version 2.

    Arguments:
        data {[json]} -- [Video Indexer Output]

    Returns:
        [Dictionary - (Keyword Name: Keyword Duration)] -- [Ranked from highest to lowest score]
    """
    keyword_names = []
    keyword_durations = []
    for keyword in data["summarizedInsights"]["keywords"]:
        keyword_names.append(keyword["name"])
        duration = 0
        for item in keyword["appearances"]:
            duration += (item["endSeconds"] - item["startSeconds"])
        keyword_durations.append(round(duration))
    keywords = dict(zip(keyword_names, keyword_durations))
    sorted_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))

    return sorted_keywords

def get_labels_v2(data):
    """Extract Labels from the Analysed Video using Video Indexer

    Arguments:
        data {[json]} -- [Video Indexer Output]

    Returns:
        [Dictionary - (Label Name: Label SeenDuration)] -- [Ranked from highest to lowest]
    """

    label_names = []
    label_durations = []
    for label in data["summarizedInsights"]["labels"]:
        label_names.append(label["name"])
        duration = 0
        for item in label["appearances"]:
            duration += (item["endSeconds"] - item["startSeconds"])
        label_durations.append(round(duration))
    labels = dict(zip(label_names, label_durations))
    sorted_labels = dict(sorted(labels.items(), key=lambda x: x[1], reverse=True))

    return sorted_labels

def get_brands_v2(data):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Returns:
        Dictionary -- [Key - Brand Name: Value - (Seen Duration (s), Confidence, Description)]
    """

    brand_name = []
    brand_duration = []
    for brand in data["summarizedInsights"]["brands"]:
        brand_name.append(brand["name"])
        brand_duration.append(round(brand["seenDuration"]))

    brands = dict(zip(brand_name, brand_duration))
    sorted_brands = dict(sorted(brands.items(), key=lambda x: x[1], reverse=True))

    return sorted_brands

def get_emotions_v2(data):
    emotion_names = []
    emotion_durations = []
    for emotion in data["summarizedInsights"]["emotions"]:
        emotion_names.append(emotion["type"])
        duration = 0
        for item in emotion["appearances"]:
            duration += (item["endSeconds"] - item["startSeconds"])
        emotion_durations.append(round(duration))
    emotions = dict(zip(emotion_names, emotion_durations))
    sorted_emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))

    return sorted_emotions


def get_sentiments_v2(data):
    sentiments = {
        "Negative": 0,
        "Neutral": 0,
        "Positive": 0
    }

    for sentiment in data["summarizedInsights"]["sentiments"]:
        for item in sentiment["appearances"]:
            sentiments[sentiment["sentimentKey"]] += round(item["endSeconds"] - item["startSeconds"])

    return sentiments

def get_audio_effects_v2(data):
    effect_names = []
    effect_durations = []
    for effect in data["summarizedInsights"]["audioEffects"]:
        effect_names.append(effect["audioEffectKey"])
        duration = 0
        for item in effect["appearances"]:
            duration += (item["endSeconds"] - item["startSeconds"])
        effect_durations.append(round(duration))
    effects = dict(zip(effect_names, effect_durations))
    sorted_effects = dict(sorted(effects.items(), key=lambda x: x[1], reverse=True))

    return sorted_effects

def get_statistics_v2(data):
    return data["summarizedInsights"]["statistics"]

def get_ocr_v2(data):
    try:
        ocr_names = []
        ocr_durations = []
        ocr_left = []
        ocr_top = []
        ocr_width = []
        ocr_height = []
        for ocr in data["videos"][0]["insights"]["ocr"]:
            ocr_names.append(ocr["text"])
            ocr_left.append(ocr["left"])
            ocr_top.append(ocr["top"])
            ocr_width.append(ocr["width"])
            ocr_height.append(ocr["height"])
            duration = 0
            for item in ocr["instances"]:
                duration += convert_to_seconds(item["end"]) - convert_to_seconds(item["start"])
            ocr_durations.append(round(duration))
        ocrs = dict(zip(ocr_names, zip(ocr_durations, ocr_left, ocr_top, ocr_width, ocr_height)))
        sorted_ocrs = dict(sorted(ocrs.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        sorted_ocrs = {}

    return sorted_ocrs

def get_transcript_v2(data):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Returns:
        [String] -- [Transcript for the whole video]
    """

    try:
        transcript = ""
        for block in data["videos"][0]["insights"]["transcript"]:
            transcript += (block["text"] + " ")
    except Exception as e:
        transcript = ""
    return transcript

def get_shots_v2(data):
    shot_id = []
    shot_duration = []
    for shot in data["videos"][0]["insights"]["shots"]:
        shot_id.append(shot["id"])
        duration = 0
        for item in shot["keyFrames"]:
            for item_ in item["instances"]:
                duration += convert_to_seconds(item_["end"]) - convert_to_seconds(item_["start"])
        shot_duration.append(round(duration))
    shots = dict(zip(shot_id, shot_duration))
    sorted_shots = dict(sorted(shots.items(), key=lambda x: x[1], reverse=True))

    return sorted_shots

def get_frame_patterns_v2(data):
    try:
        frame_names = []
        frame_durations = []
        for frame in data["videos"][0]["insights"]["framePatterns"]:
            frame_names.append(frame["patternType"])
            duration = 0
            for item in frame["instances"]:
                duration += convert_to_seconds(item["end"]) - convert_to_seconds(item["start"])
            frame_durations.append(round(duration))
        frames = dict(zip(frame_names, frame_durations))
        sorted_frames = dict(sorted(frames.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
#         print("Error: " + data["name"] + " - " + str(e))
        sorted_frames = {}

    return sorted_frames

def get_text_content_mod_v2(data):
    return data["videos"][0]["insights"]["textualContentModeration"]

def get_video_variables_v2(idx, data):
    return {
        "id": idx,
        "indexer_source_language": get_source_language_v2(data),
        "indexer_duration": get_video_duration_v2(data),
        "indexer_faces": get_faces_v2(data),
        "indexer_topics": get_topics_v2(data),
        "indexer_keywords": get_keywords_v2(data),
        "indexer_labels": get_labels_v2(data),
        "indexer_brands": get_brands_v2(data),
        "indexer_emotions": get_emotions_v2(data),
        "indexer_sentiment_positive": get_sentiments_v2(data)["Positive"],
        "indexer_sentiment_neutral": get_sentiments_v2(data)["Neutral"],
        "indexer_sentiment_negative": get_sentiments_v2(data)["Negative"],
        "indexer_audio_effects": get_audio_effects_v2(data),
        "indexer_statistics": get_statistics_v2(data),
        "indexer_ocr": get_ocr_v2(data), 
        "indexer_transcript": get_transcript_v2(data),
        "indexer_shots": get_shots_v2(data),
        "indexer_frame_patterns": get_frame_patterns_v2(data),
        "indexer_content_moderation_banned_words_count": get_text_content_mod_v2(data)["bannedWordsCount"],
        "indexer_content_moderation_banned_words_ratio": get_text_content_mod_v2(data)["bannedWordsRatio"]
    }


def create_item_dummies(df, column, num):
    print("Processing: " + str(column))
    df_copy = df.copy()

    item_counts = df_copy[column].apply(lambda d: {k: 1 for k in d.keys()})
    item_iterator = itertools.accumulate(item_counts, lambda x, y: { k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y) })
    final_item_count = {}
    for final_item_count in item_iterator:
        pass
    sorted_item_count = sorted(final_item_count.items(), key=lambda kv: kv[1], reverse=True)

    sorted_item_keys = [item[0] for item in sorted_item_count]
    for item_key in sorted_item_keys[:num]:
        new_column_name = column + "_" + item_key.lower().replace(" ", "_")
        df_copy[new_column_name] = df[column].apply(item_match, args=(item_key,))

    return df_copy


def item_match(item_dict, item_value):
    if item_value in item_dict.keys():
        return item_dict[item_value]
    else:
        return 0

def get_ocr_area(item):
    if item == 0:
        return 0
    else:
        _, _, _, width, height = item
        return width * height

def get_ocr_duration(item):
    if item == 0:
        return 0
    else:
        return item[0]

def video_indexer_to_df(json_list, video_ids=[]):
    """
    Inputs
        json_list: List of Video Indexer Data
        video_ids: [Optional] List of Video IDs

    Outputs
        Clean DataFrame representation of the json_list
    """
    if len(video_ids) == 0:
        video_ids = range(len(json_list))
    video_indexer_attributes = []
    for idx, item in zip(video_ids, json_list):
        video_indexer_attributes.append(get_video_variables_v2(idx, item))

    df_indexer = pd.DataFrame(video_indexer_attributes)

    # Play around with numerical values
    create_item_dummies_list = {
        "indexer_audio_effects": 50,
        "indexer_brands": 50,
        "indexer_emotions": 50,
        "indexer_frame_patterns": 50,
        "indexer_keywords": 50,
        "indexer_labels": 50,
        "indexer_ocr": 50,
        "indexer_topics": 50,
    }


    for key, value in create_item_dummies_list.items():
        df_indexer = create_item_dummies(df_indexer, key, value)

    for column in df_indexer.columns:
        if "indexer_ocr_" in column:
            df_indexer[column + "_area"] = df_indexer[column].apply(get_ocr_area)
            df_indexer[column] = df_indexer[column].apply(get_ocr_duration)

    df_indexer["indexer_num_faces"] = df_indexer["indexer_faces"].apply(lambda x: len(x))
    df_indexer["indexer_num_shots"] = df_indexer["indexer_shots"].apply(lambda x: len(x))
    df_indexer["indexer_transcript"] = df_indexer["indexer_transcript"].apply(lambda x: len(x.split()))
    df_indexer["indexer_statistics_correspondence_count"] = df_indexer["indexer_statistics"].apply(lambda x: x["correspondenceCount"])

    columns_remove = [
        "indexer_audio_effects",
        "indexer_brands",
        "indexer_emotions",
        "indexer_faces",
        "indexer_frame_patterns",
        "indexer_keywords",
        "indexer_labels",
        "indexer_ocr",
        "indexer_shots",
        "indexer_statistics",
        "indexer_topics"
    ]

    df_indexer_clean = df_indexer.drop(columns=columns_remove, axis=1)
    return df_indexer_clean


def variance_inflation_factor(X):
    """
    Calculates the variance inflation factor for each column in a dataframe. Credit to Daniel-Han Chan
    Input:
        X: Dataframe
    Output:
        Dataframe of variance inflation values for each column.
    """
    isDataframe = type(X) is pd.DataFrame
    if isDataframe:
        columns = X.columns
        X = X.values
    n, p = X.shape

    swap = np.arange(p)
    np.random.shuffle(swap)

    XTX = X.T @ X
    XTX = XTX[swap][:,swap]

    select = np.ones(p, dtype = bool)

    temp = XTX.copy().T
    error = 1
    largest = XTX.diagonal().max() // 2
    add = largest
    maximum = np.finfo(np.float32).max

    while error != 0:
        C, error = spotrf(a = temp)
        if error != 0:
            error -= 1
            select[error] = False
            temp[error, error] += add
            error += 1

            add += np.random.randint(1,30)
            add *= np.random.randint(30,50)
        if add > maximum:
            add = largest

    VIF = np.empty(p, dtype = np.float32)
    means = np.mean(X, axis = 0)[swap]

    for i in range(p):
        curr = select.copy()
        s = swap[i]

        if curr[i] == False:
            VIF[s] = np.inf
            continue
        curr[i] = False

        XX = XTX[curr]
        xtx = XX[:, curr]
        xty = XX[:,i]
        y_x = X[:,s]

        theta_x = sposv(xtx, xty)[1]
        y_hat = X[:,swap[curr]] @ theta_x

        SS_res = y_x-y_hat
        SS_res = np.einsum('i,i', SS_res, SS_res)
        #SS_res = np.sum((y_x - y_hat)**2)

        SS_tot = y_x - means[i]
        SS_tot = np.einsum('i,i', SS_tot, SS_tot)
        #SS_tot = np.sum((y_x - np.mean(y_x))**2)
        if SS_tot == 0:
            R2 = 1
            VIF[s] = np.inf
        else:
            R2 = 1 - (SS_res/SS_tot)
            VIF[s] = 1/(1-R2)
        del XX, xtx, xty, y_x, theta_x, y_hat
    if isDataframe:
        df_vif = pd.DataFrame({"vif": VIF})
        df_vif = df_vif.set_index(columns)
        return df_vif
    return VIF

class VideoIndexerClientV2():
    """Microsoft Video Indexer Wrapper Class"""
    def __init__(self, subscription_key, location, account_id):
        self._subscription_key = subscription_key
        self._location = location
        self._account_id = account_id

    def get_account_access_token(self, allow_edit=True):
        """Gets user account access token, which is required to make calls to APIs listed under "operation"."""

        headers = {
            'Ocp-Apim-Subscription-Key': self._subscription_key,
        }

        params = urllib.parse.urlencode({
            'allowEdit': str(allow_edit),
        })
        try:
            url = "https://api.videoindexer.ai/auth/" + self._location + "/Accounts/" + self._account_id + "/AccessToken?"
            return requests.get(url, params=params, headers=headers).json()
        except Exception as e:
            print("Get Account Access Token Error: " + str(e))
            return None

    def delete_video(self, video_id):
        """
        Deletes an uploaded Video Indexer video provided its video_id.
        NOTE: video_id is NOT the same as external_id. Use "get_video_id_by_external_id()" to get the video_id, or
        find it in the .JSON output file.
        """
        access_token = self.get_account_access_token()
        headers = {
        }

        params = urllib.parse.urlencode({
            'accessToken': access_token,
        })

        try:
            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos/" + video_id + "?"
            return requests.delete(url, params=params, headers=headers)
        except Exception as e:
            print("Delete Video Error: " + video_id + ": " + str(e))
            return None

    def list_videos(self):
        """
        Lists all the videos that are currently uploaded on Microsoft Video Indexer.
        https://www.videoindexer.ai/
        """
        access_token = self.get_account_access_token()
        headers = {
            # Request headers
            'Content-Type': 'application/json',
        }

        params = urllib.parse.urlencode({
            'accessToken': access_token,
        })

        try:
            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos?"
            return requests.get(url, params=params, headers=headers).json()
        except Exception as e:
            print("List Videos Error: " + str(e))
            return None

    def search_videos(self, external_id):
        """
        Searches for an uploaded video by its external_id.
        """
        access_token = self.get_account_access_token()
        headers = {}

        params = urllib.parse.urlencode({
            # Request parameters
            'externalId': external_id,
            'accessToken': access_token,
        })

        try:
            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos/Search?"
            return requests.get(url, params=params, headers=headers).json()
        except Exception as e:
            print("Search Videos Error: " + external_id + ": " + str(e))
            return None

    def get_video_id_by_external_id(self, external_id):
        """
        Retrieves the video_id (used for get_video_index) from the external_id.
        """
        access_token = self.get_account_access_token()
        headers = {}

        params = urllib.parse.urlencode({
            # Request parameters
            'externalId': external_id,
            'accessToken': access_token,
        })

        try:
            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos/GetIdByExternalId?"
            return requests.get(url, params=params, headers=headers).json()
        except Exception as e:
            print("Get Video Id by External Id Error: " + external_id + ": " + str(e))
            return None


    def get_video_index(self, video_id):
        """
        Retrieves the JSON output file detailing the Video Indexer's analysis results.
        See https://docs.microsoft.com/en-us/azure/media-services/video-indexer/video-indexer-output-json-v2
        """
        access_token = self.get_account_access_token()
        headers = {}

        params = urllib.parse.urlencode({
            'accessToken': access_token,
        })

        try:
            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos/" + video_id + "/Index?"
            return requests.get(url, params=params, headers=headers).json()
        except Exception as e:
            print("Get Video Index Error: " + video_id + ": " + str(e))
            return None

    def upload_video(self, name, video_url, description, external_id, privacy="Private"):
        """
        Upload a video to Video Indexer using an online url.
        """
        access_token = self.get_account_access_token()
        headers = {
            # Request headers
            'Content-Type': 'multipart/form-data',
        }

        params = urllib.parse.urlencode({
            # Request parameters
            'accessToken': access_token,
            'videoUrl': video_url,
            'name': name,
            'description': description,
            'externalId': external_id,
            'privacy': privacy,
        })

        try:
            print("Uploading Video: " + external_id)
            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos?"
            return requests.post(url, params=params, headers=headers).json()
        except Exception as e:
            print("Upload Video Error " + external_id + ": " + str(e))
            return None

    def upload_video_local(self, name, video_path, description, external_id, privacy="Private"):
        """
        Upload a video to Video Indexer using a video path local to your computer (or google colaboratory)
        """
        access_token = self.get_account_access_token()
        headers = {
            # Request headers
            'Content-Type': 'multipart/form-data',
        }

        params = urllib.parse.urlencode({
            'location' : self._location,
            'accountId' : self._account_id,
            'accessToken': access_token,
            'name': name,
            'description': description,
            'externalId': external_id,
            'privacy': privacy,
        })

        try:
            print("Uploading Video: " + name)

            url = "https://api.videoindexer.ai/" + self._location + "/Accounts/" + self._account_id + "/Videos?"

            return requests.post(url, params=params,files={'file0': open(video_path, 'rb')}).json()

        except Exception as e:
            print("Upload Video Error " + name + ": " + str(e))
            return None


    def get_faces(self, video_id):
        """
        NOTE: Used for Public Videos only
        Retrieves urls for faces detected in a video.
        """
        access_token = self.get_account_access_token()
        face_urls = []
        try:
            data = self.get_video_index(access_token=access_token, video_id=video_id)
            for item in data["summarizedInsights"]["faces"]:
                face_urls.append("https://www.videoindexer.ai/api/v2/accounts/" + self._account_id + "/videos/" + data["id"] + "/thumbnails/" + item["thumbnailId"] + "/")
            return face_urls
        except Exception as e:
            print("Get Faces Error " + str(video_id) + ": " + str(e))
            return face_urls