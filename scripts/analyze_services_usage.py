#!/usr/bin/env python3
"""
分析 services 目录下各服务的使用情况
识别冗余、过期和失效的代码
"""

import os
import re
import ast
import sys
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class ServiceAnalyzer:
    """服务使用情况分析器"""

    def __init__(self):
        self.services_dir = Path(__file__).parent.parent / "services"
        self.project_root = Path(__file__).parent.parent
        self.import_graph = defaultdict(set)
        self.service_stats = {}
        self.redundant_files = []
        self.outdated_files = []

    def analyze_service(self, service_path):
        """分析单个服务文件"""
        service_name = service_path.stem
        stats = {
            'name': service_name,
            'path': str(service_path),
            'size': service_path.stat().st_size,
            'lines': 0,
            'classes': [],
            'functions': [],
            'imports': [],
            'exported_names': [],
            'used_by': set(),
            'uses': set(),
            'is_entry_point': False,
            'is_test': False,
            'has_main': False,
            'last_modified': service_path.stat().st_mtime
        }

        try:
            with open(service_path, 'r', encoding='utf-8') as f:
                content = f.read()
                stats['lines'] = len(content.splitlines())

            # 解析 AST
            tree = ast.parse(content)

            # 提取类和函数
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    stats['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    stats['functions'].append(node.name)

            # 提取导入和导出
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        stats['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            if alias.name == '*':
                                stats['imports'].append(f"{node.module}.*")
                            else:
                                stats['imports'].append(f"{node.module}.{alias.name}")

            # 检查特殊标记
            if '__main__' in content:
                stats['has_main'] = True

            if 'test' in service_name.lower() or 'Test' in content:
                stats['is_test'] = True

        except Exception as e:
            print(f"分析文件失败 {service_path}: {e}")

        return stats

    def find_imports_in_project(self):
        """在整个项目中查找导入关系"""
        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            if file_path == Path(__file__):
                continue  # 跳过分析脚本本身

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 查找对 services 模块的导入
                pattern = r'from\s+services\.(\w+)(?:\.(\w+))?\s+import'
                matches = re.findall(pattern, content)

                for match in matches:
                    service_name = match[0]
                    imported_name = match[1] if match[1] else ''

                    # 记录依赖关系
                    if service_name in self.service_stats:
                        self.service_stats[service_name]['used_by'].add(str(file_path.relative_to(self.project_root)))

                        # 记录导出的名称
                        if imported_name and imported_name not in self.service_stats[service_name]['exported_names']:
                            self.service_stats[service_name]['exported_names'].append(imported_name)

            except Exception as e:
                print(f"处理文件失败 {file_path}: {e}")

    def identify_redundant_files(self):
        """识别冗余文件"""
        for service_name, stats in self.service_stats.items():
            # 跳过核心文件
            if service_name in ['__init__', 'unified_scheduler', 'scheduler_config']:
                continue

            # 检查是否被使用
            is_used = len(stats['used_by']) > 0
            is_scheduler_related = service_name in ['scheduler_service', 'schedule_service']
            is_agent_related = 'agent' in service_name.lower()
            is_ws_related = 'ws' in service_name.lower() or 'websocket' in service_name.lower()

            # 特殊检查：调度器服务已被统一调度器替代
            if is_scheduler_related:
                stats['redundant_reason'] = "已被 unified_scheduler 替代"
                self.redundant_files.append(stats)

            # 检查重复的 WebSocket 相关服务
            elif is_ws_related and len([s for s in self.service_stats.keys() if 'ws' in s.lower() or 'websocket' in s.lower()]) > 1:
                # 检查功能重复
                if service_name in ['ws_data_service'] and not is_used:  # 可能的重复
                    stats['redundant_reason'] = "WebSocket 数据服务可能与其他服务重复"
                    self.redundant_files.append(stats)

            # 检查未使用的文件
            elif not is_used and not stats['is_entry_point']:
                if not service_name.endswith('_service') or len(stats['classes']) == 0:
                    stats['redundant_reason'] = "未被使用且非服务类文件"
                    self.redundant_files.append(stats)

    def identify_potentially_outdated(self):
        """识别可能过时的文件"""
        current_time = Path(__file__).stat().st_mtime
        thirty_days_ago = current_time - (30 * 24 * 60 * 60)

        for service_name, stats in self.service_stats.items():
            # 跳过核心文件和已识别的冗余文件
            if service_name in ['__init__', 'unified_scheduler', 'scheduler_config']:
                continue
            if stats in self.redundant_files:
                continue

            # 检查是否长时间未修改
            if stats['last_modified'] < thirty_days_ago:
                # 检查是否包含废弃标记
                try:
                    with open(stats['path'], 'r', encoding='utf-8') as f:
                        content = f.read()

                    outdated_indicators = [
                        'DEPRECATED',
                        'TODO: remove',
                        'FIXME: obsolete',
                        '# LEGACY',
                        '# @deprecated'
                    ]

                    if any(indicator in content for indicator in outdated_indicators):
                        stats['outdated_reason'] = "包含废弃标记"
                        self.outdated_files.append(stats)

                except Exception:
                    pass

    def check_for_similar_services(self):
        """检查功能相似的服务"""
        # 检查 WebSocket 相关服务
        ws_services = [name for name in self.service_stats.keys()
                      if any(keyword in name.lower() for keyword in ['ws', 'websocket', 'connection'])]

        # 检查 Agent 相关服务
        agent_services = [name for name in self.service_stats.keys()
                         if 'agent' in name.lower()]

        # 检查数据相关服务
        data_services = [name for name in self.service_stats.keys()
                        if any(keyword in name.lower() for keyword in ['data', 'sync'])]

        return {
            'websocket_services': ws_services,
            'agent_services': agent_services,
            'data_services': data_services
        }

    def analyze(self):
        """执行完整分析"""
        print("🔍 开始分析 services 目录...")

        # 分析所有服务文件
        for service_file in self.services_dir.glob("*.py"):
            if service_file.name == "__init__.py":
                continue

            stats = self.analyze_service(service_file)
            self.service_stats[service_file.stem] = stats

        print(f"✅ 分析了 {len(self.service_stats)} 个服务文件")

        # 查找导入关系
        print("🔗 分析项目中的导入关系...")
        self.find_imports_in_project()

        # 识别冗余文件
        print("🔍 识别冗余文件...")
        self.identify_redundant_files()

        # 识别过时文件
        print("📅 识别过时文件...")
        self.identify_potentially_outdated()

        # 检查相似服务
        print("🔀 检查相似服务...")
        similar_services = self.check_for_similar_services()

        return {
            'total_services': len(self.service_stats),
            'redundant_files': self.redundant_files,
            'outdated_files': self.outdated_files,
            'similar_services': similar_services,
            'service_stats': self.service_stats
        }

    def print_report(self, results):
        """打印分析报告"""
        print("\n" + "=" * 80)
        print("📊 SERVICES 目录分析报告")
        print("=" * 80)

        print(f"\n📈 总体统计:")
        print(f"  总服务文件数: {results['total_services']}")

        print(f"\n🗑️  冗余文件 ({len(results['redundant_files'])} 个):")
        if results['redundant_files']:
            for file_info in results['redundant_files']:
                print(f"  ❌ {file_info['name']}")
                print(f"     原因: {file_info.get('redundant_reason', '未知')}")
                print(f"     大小: {file_info['size']} bytes")
                print(f"     行数: {file_info['lines']}")
        else:
            print("  ✅ 未发现冗余文件")

        print(f"\n📅 过时文件 ({len(results['outdated_files'])} 个):")
        if results['outdated_files']:
            for file_info in results['outdated_files']:
                print(f"  ⚠️  {file_info['name']}")
                print(f"     原因: {file_info.get('outdated_reason', '未知')}")
        else:
            print("  ✅ 未发现过时文件")

        print(f"\n🔀 相似功能服务:")
        for category, services in results['similar_services'].items():
            if len(services) > 1:
                print(f"  {category}: {', '.join(services)}")

        print(f"\n📋 使用情况统计:")
        unused_count = 0
        for name, stats in results['service_stats'].items():
            if not stats['used_by'] and not stats['is_entry_point'] and name not in ['__init__']:
                unused_count += 1

        print(f"  未被使用的服务: {unused_count} 个")

        if unused_count > 0:
            print("\n  未使用的服务列表:")
            for name, stats in results['service_stats'].items():
                if not stats['used_by'] and not stats['is_entry_point'] and name not in ['__init__']:
                    print(f"    - {name} ({stats['lines']} lines)")

    def generate_removal_plan(self, results):
        """生成安全的移除计划"""
        plan = {
            'safe_to_remove': [],
            'needs_review': [],
            'keep': []
        }

        for file_info in results['redundant_files']:
            # 调度器服务已被统一调度器替代，可以安全移除
            if file_info['name'] in ['scheduler_service', 'schedule_service']:
                # 等待确认后再移除
                plan['needs_review'].append({
                    'file': file_info['name'],
                    'reason': '调度器服务已被统一调度器替代',
                    'action': '移除（确认统一调度器正常工作后）',
                    'dependencies': file_info['used_by']
                })

        return plan

def main():
    """主函数"""
    analyzer = ServiceAnalyzer()
    results = analyzer.analyze()

    # 打印报告
    analyzer.print_report(results)

    # 生成移除计划
    print(f"\n" + "=" * 80)
    print("🗑️  清理建议")
    print("=" * 80)

    removal_plan = analyzer.generate_removal_plan(results)

    print(f"\n🔍 需要人工审查的文件:")
    for item in removal_plan['needs_review']:
        print(f"  📁 {item['file']}.py")
        print(f"     原因: {item['reason']}")
        print(f"     建议: {item['action']}")
        if item['dependencies']:
            print(f"     被以下文件引用: {', '.join(item['dependencies'])}")
        print()

    return results

if __name__ == "__main__":
    main()