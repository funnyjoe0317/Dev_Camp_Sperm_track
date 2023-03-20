import csv
import numpy as np


    # ------------ min_track_length -> 프레임 길이에 맞게 설정 해줘야 함----------------
def analyze_motility(tracks,  min_track_length=20):
    motility_results = {}
    sperm_counts = {}

    video_name = "sample_video"
    video_tracks = tracks

    motility_data = calculate_motility(video_tracks)
    motility_results[video_name] = motility_data
    
    # # 고유한 트랙 ID 수를 세기
    # sperm_count = len(set([track_id for track in video_tracks.values() for _, _, track_id in track]))
    # sperm_counts[video_name] = sperm_count
    # 고유한 트랙 ID 수를 세기 전에 트랙 길이 필터링
    filtered_tracks = {}
    for frame, track_list in video_tracks.items():
        for track in track_list:
            x, y, track_id = track
            if track_id not in filtered_tracks:
                filtered_tracks[track_id] = []
            filtered_tracks[track_id].append((x, y))
    
    # 최소 프레임 길이를 만족하는 트랙만 남기기
    long_tracks = {track_id: track for track_id, track in filtered_tracks.items() if len(track) >= min_track_length}

    sperm_counts[video_name] = len(long_tracks)
  
    return motility_results, sperm_counts


def calculate_motility(video_tracks):
    progressive_threshold = 10
    non_progressive_threshold = 5

    total_sperms = sum([len(track) for track in video_tracks.values()])
    progressive_count = 0
    non_progressive_count = 0
    immotile_count = 0

    for track in video_tracks.values():
        for i in range(len(track) - 1):
            current_position = np.array(track[i])
            next_position = np.array(track[i + 1])
            distance = np.linalg.norm(next_position - current_position)

            if distance >= progressive_threshold:
                progressive_count += 1
                break
            elif distance >= non_progressive_threshold:
                non_progressive_count += 1
                break
        else:
            immotile_count += 1

    motility_data = {
        "progressive": (progressive_count / total_sperms) * 100 if total_sperms > 0 else 0,
        "non_progressive": (non_progressive_count / total_sperms) * 100 if total_sperms > 0 else 0,
        "immotile": (immotile_count / total_sperms) * 100 if total_sperms > 0 else 0
    }

    return motility_data
