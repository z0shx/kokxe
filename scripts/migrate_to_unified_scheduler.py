#!/usr/bin/env python3
"""
迁移到统一调度器脚本

这个脚本将：
1. 更新所有引用旧调度器的地方
2. 提供回滚选项
3. 验证迁移结果
"""

import os
import sys
import re
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def find_files_with_old_scheduler():
    """查找使用旧调度器的文件"""
    project_root = Path(__file__).parent.parent
    patterns = [
        '*.py',
        'ui/*.py',
        'services/*.py',
        'scripts/*.py',
    ]

    files_with_old_scheduler = []

    for pattern in patterns:
        for file_path in project_root.glob(pattern):
            if file_path.name in ['migrate_to_unified_scheduler.py', 'unified_scheduler.py', 'scheduler_config.py']:
                continue  # 跳过这些文件

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否使用旧调度器
                old_patterns = [
                    r'from\s+services\.scheduler_service\s+import\s+',
                    r'from\s+services\.schedule_service\s+import\s+',
                    r'scheduler_service\.SchedulerService',
                    r'schedule_service\.ScheduleService',
                    r'\.scheduler_service\b',
                    r'\.schedule_service\b',
                ]

                for pattern in old_patterns:
                    if re.search(pattern, content):
                        files_with_old_scheduler.append({
                            'file': str(file_path.relative_to(project_root)),
                            'matches': [m.group() for m in re.finditer(pattern, content)]
                        })
                        break

            except Exception as e:
                print(f"读取文件失败 {file_path}: {e}")

    return files_with_old_scheduler

def update_imports(file_path, dry_run=True):
    """更新文件中的导入语句"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 替换导入语句
        content = re.sub(
            r'from\s+services\.scheduler_service\s+import\s+(\w+)',
            r'from services.scheduler_config import scheduler_config\nfrom services.unified_scheduler import unified_scheduler\n\1 = unified_scheduler if scheduler_config.should_use_unified_scheduler() else None',
            content
        )

        content = re.sub(
            r'from\s+services\.schedule_service\s+import\s+(\w+)',
            r'from services.scheduler_config import scheduler_config\nfrom services.unified_scheduler import unified_scheduler\n\1 = unified_scheduler if scheduler_config.should_use_unified_scheduler() else None',
            content
        )

        # 替换直接引用
        content = re.sub(r'scheduler_service\.SchedulerService', 'unified_scheduler', content)
        content = re.sub(r'schedule_service\.ScheduleService', 'unified_scheduler', content)

        # 添加兼容性检查
        if content != original_content:
            # 添加兼容性检查代码
            compatibility_check = """
# 兼容性检查：确保使用正确的调度器
try:
    from services.scheduler_config import scheduler_config
    if scheduler_config.should_use_unified_scheduler():
        from services.unified_scheduler import unified_scheduler
        scheduler_instance = unified_scheduler
    else:
        # 使用传统调度器
        from services.scheduler_service import scheduler_service
        from services.schedule_service import ScheduleService
        scheduler_instance = scheduler_service
except ImportError:
    # 回退到统一调度器
    from services.unified_scheduler import unified_scheduler
    scheduler_instance = unified_scheduler

"""

            # 在第一个导入语句后添加兼容性检查
            first_import = content.find('import')
            if first_import != -1:
                end_of_line = content.find('\n', first_import)
                if end_of_line != -1:
                    content = content[:end_of_line + 1] + compatibility_check + content[end_of_line + 1:]

        # 写回文件
        if not dry_run and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return content != original_content

    except Exception as e:
        print(f"更新文件失败 {file_path}: {e}")
        return False

def create_env_file():
    """创建环境配置文件"""
    env_path = Path(__file__).parent.parent / '.env'
    env_example_path = Path(__file__).parent.parent / '.env.example'

    # 检查 .env 文件是否存在
    if env_path.exists():
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                env_content = f.read()
        except Exception as e:
            print(f"读取 .env 文件失败: {e}")
            env_content = ""
    else:
        env_content = ""

    # 添加或更新调度器配置
    scheduler_config = """
# 调度器配置
# true: 使用统一调度器 (推荐)
# false: 使用传统调度器 (向后兼容)
USE_UNIFIED_SCHEDULER=true

# 是否禁用传统调度器
# true: 完全禁用旧调度器
# false: 保持旧调度器作为备份
DISABLE_LEGACY_SCHEDULERS=false
"""

    # 检查是否已存在配置
    if 'USE_UNIFIED_SCHEDULER' in env_content:
        print("✅ .env 文件中已存在调度器配置")
        return False

    # 添加配置到 .env 文件
    try:
        with open(env_path, 'a', encoding='utf-8') as f:
            f.write(scheduler_config)
        print("✅ 已添加调度器配置到 .env 文件")
        return True
    except Exception as e:
        print(f"写入 .env 文件失败: {e}")
        return False

def validate_migration():
    """验证迁移结果"""
    print("\n=== 验证迁移结果 ===")

    try:
        # 测试导入统一调度器
        from services.unified_scheduler import unified_scheduler
        print("✅ 统一调度器导入成功")

        # 测试导入配置
        from services.scheduler_config import scheduler_config
        print("✅ 调度器配置导入成功")

        # 显示配置信息
        config_info = scheduler_config.get_scheduler_info()
        print(f"📋 当前配置: {config_info}")

        # 测试调度器实例
        print(f"🔧 调度器实例: {type(unified_scheduler).__name__}")

        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def backup_files(file_list):
    """备份文件"""
    print("\n=== 备份文件 ===")

    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(__file__).parent.parent / f'backup_scheduler_migration_{timestamp}'
    backup_dir.mkdir(exist_ok=True)

    backed_up = []
    for file_info in file_list:
        file_path = Path(__file__).parent.parent / file_info['file']
        if file_path.exists():
            backup_path = backup_dir / file_info['file']
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            backed_up.append(file_info['file'])
            print(f"📋 备份: {file_info['file']}")

    print(f"✅ 已备份 {len(backed_up)} 个文件到: {backup_dir}")
    return backup_dir

def main():
    """主函数"""
    print("🚀 开始迁移到统一调度器...")
    print("=" * 60)

    # 1. 查找需要更新的文件
    print("\n=== 查找使用旧调度器的文件 ===")
    files_to_update = find_files_with_old_scheduler()

    if not files_to_update:
        print("✅ 没有找到使用旧调度器的文件")
        return

    print(f"📋 找到 {len(files_to_update)} 个文件需要更新:")
    for file_info in files_to_update:
        print(f"  - {file_info['file']}")
        for match in file_info['matches']:
            print(f"    匹配: {match}")

    # 2. 创建环境配置
    print("\n=== 配置环境变量 ===")
    env_updated = create_env_file()

    # 3. 备份文件
    backup_dir = backup_files(files_to_update)

    # 4. 更新文件 (dry run)
    print("\n=== 预览文件更新 (Dry Run) ===")
    updated_files = []
    for file_info in files_to_update:
        file_path = Path(__file__).parent.parent / file_info['file']
        if update_imports(file_path, dry_run=True):
            updated_files.append(file_info['file'])
            print(f"🔄 将更新: {file_info['file']}")

    if not updated_files:
        print("✅ 没有文件需要更新")
        return

    # 5. 确认更新
    print(f"\n=== 确认更新 ===")
    print(f"将更新 {len(updated_files)} 个文件")
    response = input("是否继续? (y/N): ").strip().lower()

    if response != 'y':
        print("❌ 用户取消操作")
        return

    # 6. 执行更新
    print("\n=== 执行文件更新 ===")
    success_count = 0
    for file_info in files_to_update:
        file_path = Path(__file__).parent.parent / file_info['file']
        if update_imports(file_path, dry_run=False):
            success_count += 1
            print(f"✅ 已更新: {file_info['file']}")
        else:
            print(f"❌ 更新失败: {file_info['file']}")

    print(f"\n📊 更新结果: {success_count}/{len(updated_files)} 个文件成功更新")

    # 7. 验证迁移
    print("\n=== 验证迁移 ===")
    if validate_migration():
        print("🎉 迁移成功完成!")
        print(f"📁 备份位置: {backup_dir}")
        print("\n📝 后续步骤:")
        print("1. 重启应用以使用新的统一调度器")
        print("2. 监控日志确保调度器正常工作")
        print("3. 如有问题，可以从备份恢复")
    else:
        print("❌ 迁移验证失败")
        print(f"📁 备份位置: {backup_dir}")
        print("请检查错误并手动修复")

if __name__ == "__main__":
    main()