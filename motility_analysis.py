import numpy as np



def analyze_motility(tracks, min_track_length=20):
    motility_results = {}
    sperm_counts = {}  # 정자의 개수를 저장할 딕셔너리 추가

    video_name = "sample_video"
    video_tracks = tracks

    # 트랙 길이 필터링을 위한 코드 추가
    filtered_tracks = {}
    for frame, track_list in video_tracks.items():
        for track in track_list:
            x, y, track_id = track
            if track_id not in filtered_tracks:
                filtered_tracks[track_id] = []
            filtered_tracks[track_id].append((x, y))
    
    # 최소 프레임 길이를 만족하는 트랙만 남기기
    long_tracks = {track_id: track for track_id, track in filtered_tracks.items() if len(track) >= min_track_length}


    motility_data = calculate_motility(long_tracks)  # 수정된 long_tracks 전달
    motility_results[video_name] = motility_data

    # 정자의 개수를 저장
    sperm_counts[video_name] = len(long_tracks)
    
    return motility_results, sperm_counts  # 수정된 결과 반환

def calculate_motility(long_tracks):
    progressive_threshold = 30
    non_progressive_threshold = 20

    total_sperms = len(long_tracks)
    progressive_count = 0
    non_progressive_count = 0
    immotile_count = 0

    for track in long_tracks.values():
        motility_type = calculate_total_distance(track, progressive_threshold, non_progressive_threshold)
        if motility_type == 'progressive':
            progressive_count += 1
        elif motility_type == 'non_progressive':
            non_progressive_count += 1
        else:
            immotile_count += 1

    motility_data = {
        "progressive": (progressive_count / total_sperms) * 100 if total_sperms > 0 else 0,
        "non_progressive": (non_progressive_count / total_sperms) * 100 if total_sperms > 0 else 0,
        "immotile": (immotile_count / total_sperms) * 100 if total_sperms > 0 else 0,
    }
    return motility_data



def calculate_total_distance(track, progressive_threshold, non_progressive_threshold):
    total_distance = 0
    for i in range(len(track) - 1):
        current_position = np.array(track[i])
        next_position = np.array(track[i + 1])
        distance = np.linalg.norm(next_position - current_position)
        total_distance += distance

    if total_distance >= progressive_threshold:
        return 'progressive'
    elif total_distance >= non_progressive_threshold:
        return 'non_progressive'
    else:
        return 'immotile'
