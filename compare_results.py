"""
여러 GPU Utilization 결과를 비교하는 유틸리티 스크립트
"""

import os
import json
import glob
import re
import statistics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def extract_gpu_utilization_from_folder(folder_name: str) -> Optional[float]:
    """
    폴더명에서 GPU Utilization 값을 추출
    예: 'gpu0.4' -> 0.4, 'gpu0.7' -> 0.7
    """
    match = re.search(r'gpu([\d.]+)', folder_name.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def load_result_json(result_path: str) -> Optional[Dict]:
    """JSON 결과 파일 로드"""
    json_path = os.path.join(result_path, "stt_load_test_results.json")
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 결과 파일 로드 실패 ({json_path}): {e}")
        return None


def collect_all_results(result_dir: str = "result") -> List[Tuple[float, Dict]]:
    """
    result 폴더의 모든 결과 수집
    Returns: [(gpu_utilization, result_data), ...] 리스트 (GPU Utilization 순으로 정렬됨)
    """
    if not os.path.exists(result_dir):
        print(f"⚠️ {result_dir} 폴더가 존재하지 않습니다.")
        return []
    
    results = []
    
    # result 폴더 내의 모든 하위 폴더 탐색
    for item in os.listdir(result_dir):
        folder_path = os.path.join(result_dir, item)
        if os.path.isdir(folder_path):
            gpu_util = extract_gpu_utilization_from_folder(item)
            if gpu_util is not None:
                result_data = load_result_json(folder_path)
                if result_data:
                    results.append((gpu_util, result_data))
                    print(f"✓ {item} (GPU Utilization: {gpu_util:.1%}) 로드 완료")
    
    # GPU Utilization 순으로 정렬
    results.sort(key=lambda x: x[0])
    return results


def save_comparison_graphs(results: List[Tuple[float, Dict]], output_dir: str = "result/comparison"):
    """
    여러 GPU Utilization 결과를 비교하는 그래프 생성
    """
    if not results:
        print("⚠️ 비교할 결과가 없습니다.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # GPU Utilization별 색상 맵 생성
    gpu_utils = [gpu_util for gpu_util, _ in results]
    colors = plt.cm.viridis([(u - min(gpu_utils)) / (max(gpu_utils) - min(gpu_utils)) if max(gpu_utils) > min(gpu_utils) else 0.5 for u in gpu_utils])
    
    # === 1. RTF 비교 그래프 ===
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for idx, (gpu_util, result_data) in enumerate(results):
        user_count_data = result_data.get("user_count_rtf_data", [])
        if not user_count_data:
            continue
        
        # Ramp-up과 Hold 단계 데이터 모두 사용
        ramp_up_data = [d for d in user_count_data if d.get("phase", "").startswith("ramp_up")]
        hold_data = [d for d in user_count_data if d.get("phase") == "hold"]
        
        # 모든 데이터를 사용자 수 순으로 정렬
        all_data = sorted(ramp_up_data + hold_data, key=lambda x: x.get("user_count", 0))
        
        user_counts = [d["user_count"] for d in all_data]
        avg_rtfs = [d["avg_rtf"] for d in all_data]
        
        if user_counts and avg_rtfs:
            # Ramp-up 데이터와 Hold 데이터를 다른 스타일로 표시
            ramp_up_user_counts = [d["user_count"] for d in ramp_up_data]
            ramp_up_avg_rtfs = [d["avg_rtf"] for d in ramp_up_data]
            hold_user_counts = [d["user_count"] for d in hold_data]
            hold_avg_rtfs = [d["avg_rtf"] for d in hold_data]
            
            # Ramp-up 데이터는 점선으로
            if ramp_up_user_counts and ramp_up_avg_rtfs:
                ax.plot(ramp_up_user_counts, ramp_up_avg_rtfs, '--', linewidth=1.5,
                       color=colors[idx], alpha=0.5)
            
            # Hold 데이터는 실선으로
            if hold_user_counts and hold_avg_rtfs:
                ax.plot(hold_user_counts, hold_avg_rtfs, 'o-', linewidth=2.5, markersize=10,
                       color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%} (avg RTF: {statistics.mean(hold_avg_rtfs if hold_avg_rtfs else ramp_up_avg_rtfs):.3f})')
            elif ramp_up_user_counts and ramp_up_avg_rtfs:
                # Hold 데이터가 없으면 ramp-up만 표시
                ax.plot(ramp_up_user_counts, ramp_up_avg_rtfs, 'o-', linewidth=2.5, markersize=10,
                       color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%} (avg RTF: {statistics.mean(ramp_up_avg_rtfs):.3f})')
    
    # RTF = 1.0 기준선
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
              label='RTF = 1.0 (Real-time)')
    
    ax.set_xlabel('Concurrent Users', fontsize=12)
    ax.set_ylabel('Average RTF (Real-Time Factor)', fontsize=12)
    ax.set_title('RTF Comparison by GPU Utilization', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    rtf_path = os.path.join(output_dir, "rtf_comparison.png")
    plt.savefig(rtf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 RTF 비교 그래프 저장: {rtf_path}")
    
    # === 2. Throughput 비교 그래프 ===
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for idx, (gpu_util, result_data) in enumerate(results):
        user_count_data = result_data.get("user_count_rtf_data", [])
        if not user_count_data:
            continue
        
        # Ramp-up과 Hold 단계 데이터 모두 사용
        ramp_up_data = [d for d in user_count_data if d.get("phase", "").startswith("ramp_up")]
        hold_data = [d for d in user_count_data if d.get("phase") == "hold"]
        
        # Ramp-up 데이터는 점선으로
        ramp_up_user_counts = [d["user_count"] for d in ramp_up_data]
        ramp_up_throughputs = [d.get("throughput", 0.0) for d in ramp_up_data]
        
        if ramp_up_user_counts and ramp_up_throughputs:
            ax.plot(ramp_up_user_counts, ramp_up_throughputs, '--', linewidth=1.5,
                   color=colors[idx], alpha=0.5)
        
        # Hold 데이터는 실선으로
        hold_user_counts = [d["user_count"] for d in hold_data]
        hold_throughputs = [d.get("throughput", 0.0) for d in hold_data]
        
        if hold_user_counts and hold_throughputs:
            ax.plot(hold_user_counts, hold_throughputs, 'o-', linewidth=2.5, markersize=10,
                   color=colors[idx], alpha=0.8, 
                   label=f'GPU {gpu_util:.0%} (avg: {statistics.mean(hold_throughputs):.2f} req/s)')
        elif ramp_up_user_counts and ramp_up_throughputs:
            # Hold 데이터가 없으면 ramp-up만 표시
            ax.plot(ramp_up_user_counts, ramp_up_throughputs, 'o-', linewidth=2.5, markersize=10,
                   color=colors[idx], alpha=0.8, 
                   label=f'GPU {gpu_util:.0%} (avg: {statistics.mean(ramp_up_throughputs):.2f} req/s)')
    
    ax.set_xlabel('Concurrent Users', fontsize=12)
    ax.set_ylabel('Throughput (requests/second)', fontsize=12)
    ax.set_title('Throughput Comparison by GPU Utilization', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    throughput_path = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(throughput_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Throughput 비교 그래프 저장: {throughput_path}")
    
    # === 3. TTFT 비교 그래프 (TTFT 데이터가 있는 경우) ===
    has_ttft_data = False
    for _, result_data in results:
        detailed_results = result_data.get("detailed_results", [])
        if any(r.get("ttft") is not None for r in detailed_results):
            has_ttft_data = True
            break
    
    if has_ttft_data:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        for idx, (gpu_util, result_data) in enumerate(results):
            # 동시 사용자 수별 TTFT 데이터 수집
            detailed_results = result_data.get("detailed_results", [])
            user_ttft_data: Dict[int, List[float]] = {}
            
            for r in detailed_results:
                if r.get("success") and r.get("ttft") is not None:
                    user_count = r.get("concurrent_users", 0)
                    if user_count not in user_ttft_data:
                        user_ttft_data[user_count] = []
                    user_ttft_data[user_count].append(r["ttft"])
            
            if not user_ttft_data:
                continue
            
            # Hold 단계의 최대 사용자 수 데이터 사용
            hold_results = [r for r in detailed_results 
                          if r.get("success") and r.get("ttft") is not None]
            if not hold_results:
                continue
            
            # 동시 사용자 수별로 그룹화하여 평균 계산
            sorted_users = sorted(user_ttft_data.keys())
            avg_ttfts = []
            user_counts = []
            
            for user_count in sorted_users:
                ttft_values = user_ttft_data[user_count]
                if ttft_values:
                    avg_ttfts.append(statistics.mean(ttft_values))
                    user_counts.append(user_count)
            
            if user_counts and avg_ttfts:
                ax.plot(user_counts, avg_ttfts, 'o-', linewidth=2.5, markersize=10,
                       color=colors[idx], alpha=0.8,
                       label=f'GPU {gpu_util:.0%} (avg TTFT: {statistics.mean(avg_ttfts):.3f}s)')
        
        ax.set_xlabel('Concurrent Users', fontsize=12)
        ax.set_ylabel('Average TTFT (Time to First Token) (seconds)', fontsize=12)
        ax.set_title('TTFT Comparison by GPU Utilization', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        ttft_path = os.path.join(output_dir, "ttft_comparison.png")
        plt.savefig(ttft_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 TTFT 비교 그래프 저장: {ttft_path}")
    
    # === 4. 통합 비교 그래프 (RTF, Throughput, TTFT를 하나의 figure로) ===
    num_subplots = 3 if has_ttft_data else 2
    fig, axes = plt.subplots(num_subplots, 1, figsize=(14, 6 * num_subplots))
    if num_subplots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    
    # RTF
    for idx, (gpu_util, result_data) in enumerate(results):
        user_count_data = result_data.get("user_count_rtf_data", [])
        if not user_count_data:
            continue
        
        ramp_up_data = [d for d in user_count_data if d.get("phase", "").startswith("ramp_up")]
        hold_data = [d for d in user_count_data if d.get("phase") == "hold"]
        
        # Ramp-up 데이터는 점선으로
        if ramp_up_data:
            ramp_up_user_counts = [d["user_count"] for d in ramp_up_data]
            ramp_up_avg_rtfs = [d["avg_rtf"] for d in ramp_up_data]
            if ramp_up_user_counts and ramp_up_avg_rtfs:
                ax1.plot(ramp_up_user_counts, ramp_up_avg_rtfs, '--', linewidth=1.5,
                        color=colors[idx], alpha=0.5)
        
        # Hold 데이터는 실선으로
        hold_user_counts = [d["user_count"] for d in hold_data]
        hold_avg_rtfs = [d["avg_rtf"] for d in hold_data]
        
        if hold_user_counts and hold_avg_rtfs:
            ax1.plot(hold_user_counts, hold_avg_rtfs, 'o-', linewidth=2.5, markersize=8,
                    color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%}')
        elif ramp_up_data:
            ramp_up_user_counts = [d["user_count"] for d in ramp_up_data]
            ramp_up_avg_rtfs = [d["avg_rtf"] for d in ramp_up_data]
            if ramp_up_user_counts and ramp_up_avg_rtfs:
                ax1.plot(ramp_up_user_counts, ramp_up_avg_rtfs, 'o-', linewidth=2.5, markersize=8,
                        color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%}')
    
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='RTF = 1.0 (Real-time)')
    ax1.set_xlabel('Concurrent Users', fontsize=11)
    ax1.set_ylabel('Average RTF', fontsize=11)
    ax1.set_title('RTF Comparison by GPU Utilization', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Throughput
    for idx, (gpu_util, result_data) in enumerate(results):
        user_count_data = result_data.get("user_count_rtf_data", [])
        if not user_count_data:
            continue
        
        ramp_up_data = [d for d in user_count_data if d.get("phase", "").startswith("ramp_up")]
        hold_data = [d for d in user_count_data if d.get("phase") == "hold"]
        
        # Ramp-up 데이터는 점선으로
        if ramp_up_data:
            ramp_up_user_counts = [d["user_count"] for d in ramp_up_data]
            ramp_up_throughputs = [d.get("throughput", 0.0) for d in ramp_up_data]
            if ramp_up_user_counts and ramp_up_throughputs:
                ax2.plot(ramp_up_user_counts, ramp_up_throughputs, '--', linewidth=1.5,
                        color=colors[idx], alpha=0.5)
        
        # Hold 데이터는 실선으로
        hold_user_counts = [d["user_count"] for d in hold_data]
        hold_throughputs = [d.get("throughput", 0.0) for d in hold_data]
        
        if hold_user_counts and hold_throughputs:
            ax2.plot(hold_user_counts, hold_throughputs, 'o-', linewidth=2.5, markersize=8,
                    color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%}')
        elif ramp_up_data:
            ramp_up_user_counts = [d["user_count"] for d in ramp_up_data]
            ramp_up_throughputs = [d.get("throughput", 0.0) for d in ramp_up_data]
            if ramp_up_user_counts and ramp_up_throughputs:
                ax2.plot(ramp_up_user_counts, ramp_up_throughputs, 'o-', linewidth=2.5, markersize=8,
                        color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%}')
    
    ax2.set_xlabel('Concurrent Users', fontsize=11)
    ax2.set_ylabel('Throughput (req/s)', fontsize=11)
    ax2.set_title('Throughput Comparison by GPU Utilization', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # TTFT (있는 경우)
    if has_ttft_data:
        for idx, (gpu_util, result_data) in enumerate(results):
            detailed_results = result_data.get("detailed_results", [])
            user_ttft_data: Dict[int, List[float]] = {}
            
            for r in detailed_results:
                if r.get("success") and r.get("ttft") is not None:
                    user_count = r.get("concurrent_users", 0)
                    if user_count not in user_ttft_data:
                        user_ttft_data[user_count] = []
                    user_ttft_data[user_count].append(r["ttft"])
            
            if not user_ttft_data:
                continue
            
            sorted_users = sorted(user_ttft_data.keys())
            avg_ttfts = []
            user_counts = []
            
            for user_count in sorted_users:
                ttft_values = user_ttft_data[user_count]
                if ttft_values:
                    avg_ttfts.append(statistics.mean(ttft_values))
                    user_counts.append(user_count)
            
            if user_counts and avg_ttfts:
                ax3.plot(user_counts, avg_ttfts, 'o-', linewidth=2.5, markersize=8,
                        color=colors[idx], alpha=0.8, label=f'GPU {gpu_util:.0%}')
        
        ax3.set_xlabel('Concurrent Users', fontsize=11)
        ax3.set_ylabel('Average TTFT (s)', fontsize=11)
        ax3.set_title('TTFT Comparison by GPU Utilization', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, "combined_comparison.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 통합 비교 그래프 저장: {combined_path}")
    
    # === 5. GPU Utilization별 요약 통계 출력 ===
    print("\n" + "="*60)
    print("📈 GPU Utilization별 요약 통계")
    print("="*60)
    
    for gpu_util, result_data in results:
        metrics = result_data.get("metrics", {})
        user_count_data = result_data.get("user_count_rtf_data", [])
        hold_data = [d for d in user_count_data if d.get("phase") == "hold"]
        
        print(f"\nGPU Utilization: {gpu_util:.0%}")
        print(f"  평균 RTF: {metrics.get('avg_rtf', 0):.3f}")
        print(f"  평균 처리량: {metrics.get('requests_per_second', 0):.2f} req/s")
        print(f"  평균 응답 시간: {metrics.get('avg_response_time', 0):.3f}s")
        if hold_data:
            print(f"  최대 부하 시 동시 사용자 수: {hold_data[0].get('user_count', 0)}")
            print(f"  최대 부하 시 평균 RTF: {statistics.mean([d['avg_rtf'] for d in hold_data]):.3f}")
            print(f"  최대 부하 시 평균 처리량: {statistics.mean([d.get('throughput', 0) for d in hold_data]):.2f} req/s")
    
    print("\n" + "="*60)


def main():
    """메인 함수"""
    print("🔍 result 폴더에서 결과 파일 수집 중...\n")
    
    results = collect_all_results("result")
    
    if not results:
        print("❌ 비교할 결과 파일을 찾을 수 없습니다.")
        print("   result 폴더에 'gpu0.X' 형식의 폴더가 있고, 그 안에 'stt_load_test_results.json' 파일이 있어야 합니다.")
        return
    
    print(f"\n✓ 총 {len(results)}개의 결과를 찾았습니다.\n")
    
    print("📊 비교 그래프 생성 중...\n")
    save_comparison_graphs(results)
    
    print("\n✅ 비교 완료!")


if __name__ == "__main__":
    main()

