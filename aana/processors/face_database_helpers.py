from collections import defaultdict


def parse_faceresults_images(faceid_output, bboxes_per_frame, accept_bad_faces=False):
    """Parse the output of the faceid model.

    Parameters
    ----------
    faceid_output : dict
        Output of the faceid model.
    accept_bad_faces : bool, optional
        If True, accept faces with low quality. The default is False.

    Returns:
    -------
    identities_timestamped : dict
        Dictionary with identities as keys and a dict with timestamps, bbox_xyxy, distance, face_norm and quality as values.
    """
    identities_per_frame = []
    for frame_id, frame_result in enumerate(faceid_output):
        if frame_result == "No faces identified":
            identities_per_frame.append("No faces identified")
        else:
            ids_curr_frame = []
            for face_number, face_ in enumerate(frame_result):
                if face_["person_name"] != "unknown" and (
                    face_["quality"] == "good" or accept_bad_faces
                ):
                    ids_curr_frame.append(
                        {
                            "person_name": face_["person_name"],
                            "image_id": face_["image_id"],
                            "distance": face_["distance"],
                            "norm": face_["norm"],
                            "quality": face_["quality"],
                            "bbox_xyxy": bboxes_per_frame[frame_id][face_number][0:4],
                        }
                    )
            if len(ids_curr_frame) > 0:
                identities_per_frame.append(ids_curr_frame)
            else:
                identities_per_frame.append("No faces identified")

    return identities_per_frame


def parse_faceresults_video(faceid_output, accept_bad_faces=False):
    """Parse the output of the faceid model.

    Parameters
    ----------
    faceid_output : dict
        Output of the faceid model.
    accept_bad_faces : bool, optional
        If True, accept faces with low quality. The default is False.

    Returns:
    -------
    identities_timestamped : dict
        Dictionary with identities as keys and a dict with timestamps, bbox_xyxy, distance, face_norm and quality as values.
    """
    identities_timestamped_dict = defaultdict(list)
    for frame_id, frame_result in enumerate(
        faceid_output["identified_faces_per_frame"]
    ):
        if frame_result == "No faces identified":
            continue
        else:
            for face_number, face_ in enumerate(frame_result):
                if face_["person_name"] != "unknown" and (
                    face_["quality"] == "good" or accept_bad_faces
                ):
                    identities_timestamped_dict[face_["person_name"]].append(
                        {
                            "timestamp": faceid_output["timestamps"][frame_id],
                            "bbox_xyxy": faceid_output["bboxes_per_frame"][frame_id][
                                face_number
                            ][0:4],
                            "distance": face_["distance"],
                            "face_norm": face_["norm"],
                            "quality": face_["quality"],
                        }
                    )

    return identities_timestamped_dict
