import csv
import numpy as np

def analyze_motility(tracks):
    motility_results = {}
    sperm_counts = {}

    video_name = "sample_video"
    video_tracks = tracks

    motility_data = calculate_motility(video_tracks)
    motility_results[video_name] = motility_data
    
    # 고치기 전
    # sperm_count = sum([len(track) for track in video_tracks.values()])
    # sperm_counts[video_name] = sperm_count
    
    # 고유한 트랙 ID 수를 세기
    sperm_count = len(set([track_id for track in video_tracks.values() for _, _, track_id in track]))
    sperm_counts[video_name] = sperm_count

    return motility_results, sperm_counts


def calculate_motility(video_tracks):
    progressive_threshold = 10
    # progressive_threshold는 성장이라고 판단되는 최소 거리입니다.
    non_progressive_threshold = 5
    # non_progressive_threshold는 성장이라고 판단되는 최소 거리입니다.

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
